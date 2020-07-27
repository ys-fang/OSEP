const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const msg = require('./translation');
const formatMessage = require('format-message');

const menuIconURI = null;
const blockIconURI = null;

const defaultId = 'default';
let theLocale = null;

class gasoJSON {
    constructor (runtime) {
        theLocale = this._setLocale();
        this.runtime = runtime;
        // communication related
        this.comm = runtime.ioDevices.comm;
        this.session = null;
        this.runtime.registerPeripheralExtension('gasoJSON', this);
        // session callbacks
        this.reporter = null;
        this.onmessage = this.onmessage.bind(this);
        this.onclose = this.onclose.bind(this);
        this.write = this.write.bind(this);
        // string op
        this.decoder = new TextDecoder();
        this.lineBuffer = '';

        this.data = {};
	this.txt = {};
	this.txtlenght = {};
	this.googlecolumn = {};
        this.emptyObj = {
            VALUE: {}
        };
    }

    onclose () {
        this.session = null;
    }

    write (data, parser = null) {
        if (this.session) {
            return new Promise(resolve => {
                if (parser) {
                    this.reporter = {
                        parser,
                        resolve
                    };
                }
                this.session.write(data);
            });
        }
    }

    onmessage (data) {
        const dataStr = this.decoder.decode(data);
        this.lineBuffer += dataStr;
        if (this.lineBuffer.indexOf('\n') !== -1){
            const lines = this.lineBuffer.split('\n');
            this.lineBuffer = lines.pop();
            for (const l of lines) {
                if (this.reporter) {
                    const {parser, resolve} = this.reporter;
                    resolve(parser(l));
                }
            }
        }
    }


    scan () {
        this.comm.getDeviceList().then(result => {
            this.runtime.emit(this.runtime.constructor.PERIPHERAL_LIST_UPDATE, result);
        });
    }

    _setLocale () {
        let nowLocale = '';
        switch (formatMessage.setup().locale) {
        case 'zh-tw':
            nowLocale = 'zh-tw';
            break;
        default:
            nowLocale = 'en';
            break;
        }
        return nowLocale;
    }

    getInfo () {
        theLocale = this._setLocale();

        return {
            id: 'gasoJSON',
            name: msg.name[theLocale],
            color1: '#4a90e2',
            color2: '#4a90e2',
            menuIconURI: menuIconURI,
            blockIconURI: blockIconURI,
            blocks: [
                {
                    opcode: 'fetchJSON',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        url: {
                            type: ArgumentType.STRING,
                            defaultValue: 'https://'
                        }
                    },
                    text: msg.fetchJSON[theLocale]
                },
                {
                    opcode: 'onJSONReceived',
                    blockType: BlockType.HAT,
                    isEdgeActivated: false,
                    arguments: {},
                    text: msg.onJSONReceived[theLocale]
                },
		{
                    opcode: 'readEntryFromJSON',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        variable: {
                            type: ArgumentType.STRING,
                            defaultValue: 'data'
                        },
                        n: {
                            type: ArgumentType.NUMBER,
                            defaultValue: '1'
                        }
                    },
                    text: msg.readEntryFromJSON[theLocale]
                },
                {
                    opcode: 'readAttrFromJSON',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        variable: {
                            type: ArgumentType.STRING,
                            defaultValue: 'data'
                        },
                        attr: {
                            type: ArgumentType.STRING,
                            defaultValue: 'attr'
                        }
                    },
                    text: msg.readAttrFromJSON[theLocale]
                },
                {
                    opcode: 'readFromJSON',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        id: {
                            type: ArgumentType.STRING,
                            defaultValue: 'id'
                        }
                    },
                    text: msg.readFromJSON[theLocale]
                },
		{
                    opcode: 'googleJSON',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        url: {
                            type: ArgumentType.STRING,
                            defaultValue: 'https://docs.google.com/spreadsheets/d/KEY_ID/edit?usp=sharing'
                        }
                    },
                    text: msg.googleJSON[theLocale]
                },
		{
                    opcode: 'googlecolumnTEXT',
                    blockType: BlockType.REPORTER,
                    arguments: {
			variable: {
                            type: ArgumentType.STRING,
                            defaultValue: 'data'
                        },
                        n: {
                            type: ArgumentType.NUMBER,
                            defaultValue: '1'
                        },
                        column: {
                            type: ArgumentType.STRING,
                            defaultValue: 'title'
                        }
                    },
                    text: msg.googlecolumnTEXT[theLocale]
                },
		{
                    opcode: 'writeGoogleCalc',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        url: {
                            type: ArgumentType.STRING,
                            defaultValue: 'https://script.google.com/macros/s/「key」/exec'
                        },
			column1: {
			    type: ArgumentType.STRING,
                            defaultValue: ' '
			},
			column2: {
			    type: ArgumentType.STRING,
                            defaultValue: ' '
			},
			column3: {
			    type: ArgumentType.STRING,
                            defaultValue: ' '
			}
                    },
                    text: msg.writeGoogleCalc[theLocale]
                },
		{
                    opcode: 'readtextFILE',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        url: {
                            type: ArgumentType.STRING,
                            defaultValue: 'http://0.0.0.0:8601/FILE.txt'
                        }
		    },
		    text:msg.readtextFILE[theLocale]
                },
                {
                    opcode: 'readFromTEXT',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        id: {
                            type: ArgumentType.STRING,
                            defaultValue: 'id'
                        }
                    },
                    text: msg.readFromTEXT[theLocale]
                },
                {
                    opcode: 'textLENGHT',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        id: {
                            type: ArgumentType.STRING,
                            defaultValue: 'id'
                        }
                    },
                    text: msg.textLENGHT[theLocale]
                },
		{
                    opcode: 'readtxtDATA',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        variable: {
                            type: ArgumentType.STRING,
                            defaultValue: 'data'
                        },
                        n: {
                            type: ArgumentType.NUMBER,
                            defaultValue: '1'
                        }
                    },
                    text: msg.readtxtDATA[theLocale]
                },
                {
                    opcode: 'openURL',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        url: {
                            type: ArgumentType.STRING,
                            defaultValue: 'https://'
                        }
                    },
                    text: msg.openURL[theLocale]
                }
            ]
        };
    }

    fetchJSON (args) {
        const url = args.url;
        return fetch(url).then(res => {
            if (res.ok) {
                res.json().then(json => {
                    console.log("got json set", json);
                    this.data.fetched = true;
                    this.data.data = JSON.stringify(json);
                    this.runtime.startHats('gasoJSON_onJSONReceived', {});
                });
            }
        });
    }

    googleJSON (args) {
	const urlsplit = args.url.split("/");
	const aurl ='https://spreadsheets.google.com/feeds/list/'+urlsplit[5]+'/od6/public/values?alt=json';
	const url = new URL(aurl);
        return fetch(url).then(res => {
            if (res.ok) {
                res.json().then(json => {
                    console.log("got json set", json);
                    this.data.fetched = true;
                    this.data.data = JSON.stringify(json.feed.entry);
                    this.runtime.startHats('gasoJSON_onJSONReceived', {});
                });
            }
        });
    }

    isDataFetched () {
        return this.data.fetched;
    }

    isTxtFetched () {
        return this.txt.fetched;
    }

    istextLENGHTFetched () {
        return this.txtlenght.fetched;
    }

    onJSONReceived (){
        if (this.isDataFetched()) {
            console.log('got data');
            return true;
        }
    }

    readFromJSON () {
        if (this.isDataFetched()) {
            console.log('return ', this.data.data);
            return this.data.data;
        }
        return msg.readFromJSONErr[theLocale];
    }


    readEntryFromJSON (args) {
        const variable = args.variable || this.emptyObj;
        const n = args.n;
        try {
	    const parsed = JSON.parse(variable);
            const data = parsed[n - 1];
            return typeof data === 'string' ? data : JSON.stringify(data);
        } catch (err) {
            return `Error: ${err}`;
        }
    }

    readAttrFromJSON (args) {
        const variable = args.variable || this.emptyObj;
        const attr = args.attr;
        try {
            const parsed = JSON.parse(variable);
            const data = parsed[attr];
            return typeof data === 'string' ? data : JSON.stringify(data);
        } catch (err) {
            return `Error: ${err}`;
        }
    }

    readtextFILE (args) {
	const file = args.url;
	this.txt.data = fetch(file,{method:'get'}).then(response => response.text());
	this.txt.fetched = true;
    }

    readtxtDATA (args) {
        const variable = args.variable || this.emptyObj;
        const n = args.n;
        try {
	    const parsed = variable.split('\n');
	    this.txtlenght.fetched = true;
	    this.txtlenght.data = parsed.length-1;
            const data = parsed[n - 1];
            return typeof data === 'string' ? data : txt.stringify(data);
        } catch (err) {
            return `Error: ${err}`;
        }
    }


    readFromTEXT () {
        if (this.isTxtFetched()) {
            console.log('return ', this.txt.data);
            return this.txt.data;
        }
        return msg.readFromTEXTErr[theLocale];
    }

    googlecolumnTEXT(args){
	const variable = args.variable || this.emptyObj;
        const n = args.n;
	const column = "gsx$" + args.column;
        try {
	    const parsed = JSON.parse(variable);
            var data = parsed[n - 1];
            data = JSON.stringify(data);
            const a_parsed = JSON.parse(data);
            var a_data = a_parsed[column];
	    a_data = JSON.stringify(a_data);
	    const t_parsed = JSON.parse(a_data);
	    var t_data = t_parsed["$t"];
            return typeof t_data === 'string' ? t_data : JSON.stringify(t_data);
         }catch (err){
	    return `Error: ${err}`;	
	}
    }

    textLENGHT () {
        if (this.istextLENGHTFetched()) {
            console.log('return ', this.txtlenght.data);
            return this.txtlenght.data;
        }
        return msg.textLENGHTErr[theLocale];
    }

    writeGoogleCalc (args) {
	alert("test");
	const column1 = args.column1 || defaultValue;
	const column2 = args.column2 || defaultValue;
	const column3 = args.column3 || defaultValue;
	const url = args.url;
	var gurl = url + "?c1=" + column1 + "&c2=" + column2 + "&c3=" + column3;
	return fetch(gurl).then(res => {
	    if (res.ok) {
	    }
	});
    }

    openURL (args) {
	const url = args.url;
	var openurl = window.open(url);
    }

}

module.exports = gasoJSON;
