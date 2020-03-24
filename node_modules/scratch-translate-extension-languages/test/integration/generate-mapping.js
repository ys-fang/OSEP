const generateMapping = require('../../index.js');
const nock = require('nock');
const test = require('tap').test;

test('spec', t => {
    t.type(generateMapping, 'function', 'generateMapping is a function');
    t.end();
});

test('object.shape', t => {
    generateMapping(result => {
        t.same(Object.keys(result),
            ['menuMap', 'nameMap', 'scratchToGoogleMap', 'previouslySupported', 'spokenLanguages']);
        t.end();
    });
});

test('Test menu and name map contain correct items', t => {
    // The auth request the client library makes to Google translate - always return a 200.
    nock('https://accounts.google.com:443/')
        .filteringRequestBody(function () {
            return '*';
        })
        .persist() // Do the same for all of these auth requests
        .post('/o/oauth2/token', '*')
        .reply(200, {access_token: 'a token', expires_in: 3600, token_type: 'Bearer'});

    // This reply was grabbed from nock.recorder.play().  The reply is from a request for
    // languages Google will translate to from English. The request was genreated by Google's Translate
    // client library.  It is gziped so looks like gibberish.
    // TODO: Stop using a gziped reply and make it so different requests have different responses.
    const gzipReply =
     ['1f', '8b', '08', '00', '00', '00', '00', '00',
     /* max-len eslint-disable-next-line quotes*/
     /* eslint-disable-next-line */
     "02ff9599dd6e1b371085effd1482ae5c207d81de356e2c39b202232b374d8b5eccee8e96d4f267432ea54841debd941120287012cc5c185e69810fe4cc993343eacbcd62b1ec69a6e56f8b2ff5b97e7214864203e7fad53f2f5f2dbebdfadfebfa7649fbe5abef6f02f9976f7fdf273b1285bcfcf6eeebab9f63f22788712d054b414a210f29de50b29d18922024512b6798336678566de70229174e2dd9438d8c14c405815e53fe5458ca681932d8512a59b1a936604e18c88937d46608895913de768090e2862a1739a62384b9ab05e514106e2185db42214a2901caeeced8cef089a4948bf9012570e6c56d63fde4ecde72ff8b9cf8ebeec34fa1bb44bd9d6d0ce4c4d42e42644cd976f2c81b58ee7729d2ac110114e4dd853b2345f450477fd432cf624670905166f93218d6e79b3038c53a18a6e64d9e385198c57ae61963e6a8a9f31906e5de3a3b59796ded2da6044d7ef6506cf789833c417b58e4f7c96a5c78803159550bee1494114a76c5310d0a4a0f3bcb8a93973318ef27318fe290c026b92a074ad50ac4760225bb267b7593c55de2e8c42dd7c0f0aea964b1971b3a61c689ac22451653b84d7c122f0516d0da865e1e5c0fcd69ed6318c40c98e67509baae6fa1e13f74751eaa1b120f8a16ce200f432bf625db4342e863edac1a4780627b480a77b350fb0f751052ace300d7f19626ba4e0a620a14ec5b3aaa282394db8642a05e5c82e38821171ac5911de1c166633c273102b6e44d4cacf07b583b9b92faaa92c56d7da8967db0e2096e84ad6c734ec3f9226538b8af4712179083827bac9e2f8e8b3bfe00715408dfc1ea79b4b329aa43b0832789c7f2997d1b4bedccf282f650ba5beab8574d601ebadcb61e9006ca6731055aee952247c031e10551ffbc1803535531b3c25c3cec875b8a49dc0f3d1c27b7d781c5c821d0e4b6b5a5468d6f7b58cbdb33054f6971fbba7a430d8ed81a021c0adff1a4b82508d01adec57462cd7c3a41dd3d5136f293cc1e9acc13274d8b9ea07a9fa2e6643641e93ec53497a128d43be10d9570a0569ca00413f43e7a95e125d895de97ac896d861db6211f1523141cc59a2ece79b12276f2b130c3c26e38b59a2dc154379ce36cc4c2cdd01d1a13837800ca382c75f0979b5486665919a6fab61802db59e3e291c4a7c40c6bf0ca505ded66a8fca62a5f6e6f0c7da99954374619164f5342af9a94339cb79b13192bdf5086635473e25eb1a119ce1a3b3a58718e67e86d","3bf2d68911b07dedd855971533e03decce9038a033b4915d49a322a00556cdf398c86a145fe05a9e532f8e47813f873c5f5af91dcf11dac89f96e7ebb35cee1d1c763eb09387f533ccee5f26ca2f77ce70371f6dafa9973374a28f319556fe8b013491bf8bfb9edb97ffffde5c9fbedefc07189336db731c0000"];
    const headers = ['Content-Type', 'application/json; charset=UTF-8', 'Vary', 'Origin', 'Vary',
        'X-Origin', 'Vary', 'Referer', 'Content-Encoding', 'gzip'];

    nock('https://translation.googleapis.com:443')
        .persist()
        .get('/language/translate/v2/languages')
        .query({target: /.*/})
        .reply(200, gzipReply, headers);

    /* max-len eslint-disable-next-line quotes*/
    /* eslint-disable-next-line */
    const translateReply = ["1f","8b","08000000000002ffabe65250504a492c4954b252a806b281bc92a2c4bce29cc492ccfcbc62a068345854012a8ba222352524b5a204a846c937312f25b128334f4103c6d25482aaafd521c6008fccbc944cd2b404e4179594a667162b683815251667e69068637041625e657e8e8286636e6a516676a2820fd0cb48ce06d3b15c20562d1700f256f7c127010000"];
    nock('https://translation.googleapis.com:443')
        .persist()
        .post('/language/translate/v2/', /.*/)
        .reply(200, translateReply, headers);

    generateMapping(result => {
        t.type(result, 'object', 'result is an object');
        t.same(Object.keys(result),
            ['menuMap', 'nameMap', 'scratchToGoogleMap', 'previouslySupported', 'spokenLanguages']);

        t.equal(Object.keys(result.menuMap).length, 52);
        t.equal(Object.keys(result.nameMap).length, 67);

        // es-419 is and edge case because Scratch has this langauge but google doesn't.
        // It should have a key in the menu map so we know what languages to show when the Scratch Editor language is
        // es-419. Since Google doesn't know what es-419 is, no values (i.e. lists of languages you can translate to)
        // should be es-419.
        t.equals(result.scratchToGoogleMap['es-419'], 'es', 'Scratch to Google map should contain pair es-419: es');
        t.ok(result.menuMap['es-419'], 'Map contains list for es-419');
        t.ok(result.menuMap.en, 'Map contains list for en');
        t.equals(result.menuMap.en.filter(pair => pair.code === 'es-419').length, 0,
            'es-419 is not in English list of languages to translate to');
        t.equals(result.nameMap.spanish, 'es');

        // Hindi (hi) used to be supported but should no longer be in the menu map.
        // Because it was previously supported, it should be in the name map though.
        t.notEquals(result.previouslySupported.indexOf('hi'), -1, 'Hindi is in the previously supported list');
        t.notOk(result.menuMap.hi, 'Hindi is not in menu map');
        t.equals(result.menuMap.en.filter(pair => pair.code === 'hi').length, 0,
            'Hindi is not in English list of languages to translate to');
        t.equals(result.nameMap.hindi, 'hi', 'Hindi is in name map because it was previously supported');

        // Test a few basics of an langauge code that is the same for both Scratch and Google
        // and is currently supported by both.
        t.equals(result.menuMap.de.length, 48, 'German list has 48 items in it');
        t.equals(result.nameMap.german, 'de', 'Name map contains german.');

        // Test the spoken languages map
        t.equal(Object.keys(result.spokenLanguages).length, 72);
        // Check for unmatched parens
        let names = [];
        for (let i in result.spokenLanguages) {
            names.push(...(result.spokenLanguages[i].map(o => o.name)));
        }
        let unmatched = false;
        names.forEach(name => {
            let count = 0;
            count += (name.match(/\(/g) || []).length;
            count += (name.match(/\)/g) || []).length;
            if ((count % 2) === 1) {
                unmatched = true;
            }
        });
        t.notOk(unmatched, 'no unmatched parens');

        t.end();
    });
});
