const async = require('async');
const Translate = require('@google-cloud/translate');
const locales = require('scratch-l10n');
const client = new Translate({
    credentials: {
        private_key: process.env.GOOGLE_PRIVATE_KEY,
        client_email: process.env.GOOGLE_CLIENT_EMAIL
    }
});

// List of languages that the translate extension menu used to have. The extension
// launched with an outdated list of Scratch languages. This list is used to keep backwards
// compatibility with any projects that may have blocks with these selected in them before
// we removed them from the menu.
const PREVIOUSLY_SUPPORTED_LIST = ['ab', 'ms', 'be', 'eo', 'hy', 'hi', 'kn',
    'ht', 'ku', 'la', 'mk', 'ml', 'mt', 'mr', 'mn', 'my', 'nn', 'sq', 'te', 'uz'];

const SUPPORTED_LOCALES = Object.keys(locales.default).concat(PREVIOUSLY_SUPPORTED_LIST);

// Names of spoken languages, for use by the Text to Speech extension.
const SPOKEN_LANGUAGES = {
    'zh-cn': 'Chinese (Mandarin)', // distinct from the written "chinese (simplified)" and "chinese (traditional)"
    'hi': 'Hindi', // available in text to speech but not yet in supported_locales
    'pt-br': 'Portuguese (Brazilian)', // not a separate entry in the translate extension menu
    'es-419': 'Spanish (Latin American)' // not a separate entry the translate extension menu
};
const spokenLanguageKeys = Object.keys(SPOKEN_LANGUAGES);
const spokenLanguageNamesEn = Object.values(SPOKEN_LANGUAGES);

// We need to provide a custom translation into Chinese of the name for the spoken
// version of Chinese we are providing.
const CUSTOM_NAMES_FOR_MANDARIN = {
    'zh-cn': '中文',
    'zh-tw': '中文'
};

// Scratch and Google translate have different language codes for some languages. These
// maps are used to convert between them.
const scratchToGoogleMap = {
    'zh-cn': 'zh',
    'nb': 'no',
    'he': 'iw',
    'es-419': 'es',
    'pt-br': 'pt',
    'ja-hira': 'ja'
};

// Construct the reverse map of the scratch to google mapping.
const googleToScratchMap = Object.keys(scratchToGoogleMap).reduce((mem, key) => {
    mem[scratchToGoogleMap[key]] = key;
    return mem;
}, {});

/**
 * Builds a map from translated language name to language code. e.g.
 * {espanol: es, spanish: es, japanese: ja, aleman: de, ... etc.}
 * This is used by the language menu in the translate block to decide whether to
 * accept a language name dropped on top of the menu.
 * @param {object} languageMap mapping language code to a list of langauge code, name pairs we can translate to.
 * @return {object} Object mapping from a language name to language code.
 */
var buildNameToCodeMap = function (languageMap) {
    var nameMap = {};
    let codes = Object.keys(languageMap);
    for (let i = 0; i < codes.length; ++i) {
        for (let j = 0; j < languageMap[codes[i]].length; ++j) {
            // Lowercase all the language codes for ease of comparison later.
            nameMap[languageMap[codes[i]][j].name.toLowerCase()] = languageMap[codes[i]][j].code.toLowerCase();
        }
    }
    // Add the Hiragana version of Japanese in Japanese (nihongo) by hand since Google Translate
    // only gives us the kanji version.
    nameMap['にほんご'] = 'ja';
    return nameMap;
};

/**
 * Gets an individual language's language list from Google Translate and adds the result to the
 * accumulator object.
 * @param {object} acc Accumulates results from the set of transform calls to get supported languages.
 * @param {string} langCode The language code to look up.
 * @param {number} index The index into the list of langauges we're looking up.
 * @param {function} callback The function which is called after all the iteratee functions have finished.
 */
var getLanguageList = function (acc, langCode, index, callback) {
    client.getLanguages(langCode, function (err, translateObj) {
        if (err) {
            // Invalid languages happen since Scratch supports some that Google
            // translate does not.  For ones where there is a mismatch in langauge codes,
            // .e.g. es-419 and cs, we'll add them later.
            if (err.code === 400 && err.message.indexOf('language is invalid')) {
                return callback();
            }
            // Avoid unhandled rejection, and allow exiting with error status
            return async.nextTick(callback, err);
        }
        const result = [];
        // Build up the list of languages (code and name) that we can translate to.
        for (let i in translateObj) {
            if (SUPPORTED_LOCALES.indexOf(translateObj[i].code.toLowerCase()) !== -1) {
                // Lowercase all the language codes for ease of comparison later.
                translateObj[i].code = translateObj[i].code.toLowerCase();
                result.push(translateObj[i]);
            } else if (googleToScratchMap[translateObj[i].code.toLowerCase()]) {
                // If this langauge code is a Google translate one, look up the scratch
                // version and put that in the result instead.
                let copy = Object.assign({}, translateObj[i]);
                copy.code = googleToScratchMap[translateObj[i].code].toLowerCase();
                result.push(copy);
            }
        }
        acc[langCode.toLowerCase()] = result;
        // If there's a language code that differs, e.g. scratch has es-419, but
        // Google Translate has es, add that to the map as well.
        if (googleToScratchMap[langCode.toLowerCase()] &&
          !acc[googleToScratchMap[langCode.toLowerCase()].toLowerCase()]) {
            acc[googleToScratchMap[langCode.toLowerCase()].toLowerCase()] = result;
        }
        return callback();
    });
};

/**
 * Removes languages from the previously supported list from the menu map so they
 * don't show up in the translate menu's list.
 * @param {object} menuMap A map of language code to an object that contains the
 *   language code list of all
 */
var removePreviouslySupported = function (menuMap) {
    const codes = Object.keys(menuMap);
    for (let i = 0; i < codes.length; ++i) {
        const filtered = menuMap[codes[i]].filter(function (langInfo) {
            return PREVIOUSLY_SUPPORTED_LIST.indexOf(langInfo.code) === -1;
        });
        menuMap[codes[i]] = filtered;
    }

    for (let i = 0; i < PREVIOUSLY_SUPPORTED_LIST.length; ++i) {
        if (menuMap[PREVIOUSLY_SUPPORTED_LIST[i]]) {
            delete menuMap[PREVIOUSLY_SUPPORTED_LIST[i]];
        }
    }
};

/**
 * Fix a problem with some translations of language names containing parentheses,
 * where the open paren is missing. If the final character is a close paren, and
 * there is no open paren, add an open paren after the first space.
 * @param {string} item The string to fix
 * @return {string} the fixed string
 */
var fixParens = function (item) {
    const endsWithCloseParen = item[item.length - 1] === ')';
    const hasOpenParen = item.includes('(');
    if (endsWithCloseParen && !hasOpenParen){
        let fixed = item.split(' ');
        if (fixed.length > 1) {
            fixed[1] = '(' + fixed[1];
            item = fixed.join(' ');
        }
    }
    return item;
};

/**
* Gets the translations into a particular language of the names of a set of spoken languages,
* and adds these to an accumulator object.
* @param {object} acc Accumulates results from the set of transform calls.
* @param {string} langCode The language code to to translate into.
* @param {number} index The index into the list of langauges we're looking up.
* @param {function} callback The function which is called after all the iteratee functions have finished.
*/
var translateSpokenLanguageNames = function (acc, langCode, index, callback) {
    const options = {
        from: 'en',
        to: langCode
    };
    client.translate(spokenLanguageNamesEn, options,
        function (err, translation) {
            if (err) {
                // Invalid languages happen since Scratch supports some that Google
                // translate does not.  For ones where there is a mismatch in langauge codes,
                // .e.g. es-419 and cs, we'll add them later.
                if (err.code === 400 && err.message.indexOf('language is invalid')) {
                    return callback();
                }
                // Avoid unhandled rejection, and allow exiting with error status
                return async.nextTick(callback, err);
            }
            const translatedSpokenLanguageNames = translation.map((item, i) => {
                item = fixParens(item);
                return {
                    code: spokenLanguageKeys[i],
                    name: item
                };
            });
            acc[langCode.toLowerCase()] = translatedSpokenLanguageNames;
            return callback();
        });
};

/**
 * Add entries to the spoken languages map for Scratch-specific language codes.
 * @param {object} spokenLanguages An object containing names of spoken languages
 *  translated into other languages.
 */
var addScratchEntriesToSpokenLanguages = function (spokenLanguages) {
    Object.keys(scratchToGoogleMap).forEach(key => {
        if (!spokenLanguages[key]) {
            const googleKey = scratchToGoogleMap[key];
            if (googleKey) {
                spokenLanguages[key] = spokenLanguages[googleKey];
            }
        }
    });
};

/**
 * Modify the entries in the spoken languages map for Chinese languages, to use
 * custom names for spoken Chinese (instead of the google translate version).
 * @param {object} spokenLanguages An object containing names of spoken languages
 *  translated into other languages.
 */
var useCustomChineseNames = function (spokenLanguages) {
    Object.keys(CUSTOM_NAMES_FOR_MANDARIN).forEach(key => {
        if (spokenLanguages[key]) {
            const customName = CUSTOM_NAMES_FOR_MANDARIN[key];
            const langObj = spokenLanguages[key];
            const cnObj = langObj.find(lang => lang.code === 'zh-cn');
            if (cnObj) {
                cnObj.name = customName;
            }
        }
    });
};

/**
 * Builds up an object containing information about language codes and language names.
 * menuMap is a mapping from a scratch language code to a list of languges to show in the Google Translate menu.
 * nameMap is a mapping from language names (translated into lots of lanuages) to language code.
 * scratchToGoogleMap is a mapping from Scratch language codes to Google langauge codes.
 * previouslySupported is a list of language codes that we used to put in the language list for the translate block
     but no longer do.
 * spokenLanguages is a mapping from scratch language code to a list of spoken language names that are distinct from
 *   written language names, for use in the Text to Speech extension's language menu.
 * @param {function} callback Function called with the result when building all the maps finishes.
 */
const generateMapping = module.exports = function (callback) {

    // the spokenLanguageNameMap is generated by translation requests, but we need
    // to seed the English data, because the translation from English to English does
    // not provide any results. Only this one name is needed, because it is the only
    // name in English that differs from the names of written language provided in the menuMap.
    const spokenLanguageNameMap = {
        en: [
            {
                code: 'zh-cn',
                name: 'Chinese (Mandarin)'
            }
        ]
    };

    // First, translate the spoken language names into each language.
    async.transform(
        SUPPORTED_LOCALES, spokenLanguageNameMap, translateSpokenLanguageNames
    ).then(spokenLanguages => {

        addScratchEntriesToSpokenLanguages(spokenLanguages);
        useCustomChineseNames(spokenLanguages);

        // Then, generate the full menuMap
        async.transform(SUPPORTED_LOCALES, {}, getLanguageList,
            function (err, result) {
                if (err) {
                    throw new Error(err);
                }
                // Result is a single element list containing a map from langauge code
                // to the lang code/name pairs we can translate to. e.g.
                const nameToLanguageCode = buildNameToCodeMap(result);
                // After we build the language code name map, we remove languages that used
                // to be in the list but aren't now.  We want those languges to be in the name
                // map so that if someone drops a block into the menu it still works.
                // For example, a block with value esperanto dropped into the language menu should
                // continue working even though esperanto isn't in the list anymore.
                removePreviouslySupported(result);
                const finalObject = {menuMap: result,
                    nameMap: nameToLanguageCode,
                    scratchToGoogleMap: scratchToGoogleMap,
                    previouslySupported: PREVIOUSLY_SUPPORTED_LIST,
                    spokenLanguages: spokenLanguages
                };
                callback(finalObject);
            });
    });
};

if (require.main === module) {
    generateMapping(result => {
        process.stdout.write(JSON.stringify(result));
    });
}
