import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";

/**
 * @param {Function} callback
 * @param {number | undefined} delay
 * @returns {Function}
 */
function debounce(callback, delay) {
    let timeoutId = null;
    return (...args) => {
        window.clearTimeout(timeoutId);
        timeoutId = window.setTimeout(() => {
            callback(...args);
        }, delay);
    };
}

/**
 * @param {string} url
 * @param {any} [options=undefined]
 * @returns {Promise}
 */
function request(url, options = undefined) {
    return new Promise((resolve, reject) => {
        api.fetchApi(url, options)
            .then((response) => response.json())
            .then(resolve)
            .catch(reject);
    });
}

const modelNodeType = {
    "checkpoints": "CheckpointLoaderSimple",
    "clip": "CLIPLoader",
    "clip_vision": "CLIPVisionLoader",
    "controlnet": "ControlNetLoader",
    "diffusers": "DiffusersLoader",
    "embeddings": "Embedding",
    "gligen": "GLIGENLoader",
    "hypernetworks": "HypernetworkLoader",
    "photomaker": "PhotoMakerLoader",
    "loras": "LoraLoader",
    "style_models": "StyleModelLoader",
    "unet": "UNETLoader",
    "upscale_models": "UpscaleModelLoader",
    "vae": "VAELoader",
    "vae_approx": undefined,
};

const DROPDOWN_DIRECTORY_SELECTION_CLASS = "search-dropdown-selected";

const MODEL_SORT_DATE_CREATED = "dateCreated";
const MODEL_SORT_DATE_MODIFIED = "dateModified";
const MODEL_SORT_DATE_NAME = "name";

const MODEL_EXTENSIONS = [".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"]; // TODO: ask server for?
const IMAGE_EXTENSIONS = [".apng", ".gif", ".jpeg", ".jpg", ".png", ".webp"]; // TODO: ask server for?

/**
 * @param {string | undefined} [searchPath=undefined]
 * @param {string | undefined} [dateImageModified=undefined]
 *
 * @returns {string}
 */
function imageUri(imageSearchPath = undefined, dateImageModified = undefined) {
    const path = imageSearchPath ?? "no-preview";
    const date = dateImageModified;
    let uri = `/model-manager/preview/get?uri=${path}`;
    if (date !== undefined && date !== null) {
        uri += `&v=${date}`;
    }
    return uri;
}

/**
 * Tries to return the related ComfyUI model directory if unambigious.
 *
 * @param {string | undefined} modelType - Model type.
 * @param {string | undefined} [fileType] - File type. Relevant for "Diffusers".
 *
 * @returns {(string | null)} Logical base directory name for model type. May be null if the directory is ambiguous or not a model type.
 */
function modelTypeToComfyUiDirectory(modelType, fileType) {
    if (fileType !== undefined && fileType !== null) {
        const f = fileType.toLowerCase();
        if (f == "diffusers") { return "diffusers"; } // TODO: is this correct?
    }

    if (modelType !== undefined && modelType !== null) {
        const m = modelType.toLowerCase();
        // TODO: somehow allow for SERVER to set dir?
        // TODO: allow user to choose EXISTING folder override/null? (style_models, HuggingFace) (use an object/map instead so settings can be dynamically set)
        if (m == "aestheticGradient") { return null; }
        else if (m == "checkpoint" || m == "checkpoints") { return "checkpoints"; }
        //else if (m == "") { return "clip"; }
        //else if (m == "") { return "clip_vision"; }
        else if (m == "controlnet") { return "controlnet"; }
        //else if (m == "Controlnet") { return "style_models"; } // are these controlnets? (TI-Adapter)
        //else if (m == "") { return "gligen"; }
        else if (m == "hypernetwork" || m == "hypernetworks") { return "hypernetworks"; }
        else if (m == "lora" || m == "loras") { return "loras"; }
        else if (m == "locon") { return "loras"; }
        else if (m == "motionmodule") { return null; }
        else if (m == "other") { return null; }
        else if (m == "pose") { return null; }
        else if (m == "textualinversion" || m == "embedding" || m == "embeddings") { return "embeddings"; }
        //else if (m == "") { return "unet"; }
        else if (m == "upscaler" || m == "upscale_model" || m == "upscale_models") { return "upscale_models"; }
        else if (m == "vae") { return "vae"; }
        else if (m == "wildcard" || m == "wildcards") { return null; }
        else if (m == "workflow" || m == "workflows") { return null; }
    }
    return null;
}

/**
 * Get model info from Civitai.
 *
 * @param {string} id - Model ID.
 * @param {string} apiPath - Civitai request subdirectory. "models" for 'model' urls. "model-version" for 'api' urls.
 *
 * @returns {Object} Dictionary containing recieved model info. Returns an empty if fails.
 */
async function civitai_requestInfo(id, apiPath) {
    const url = "https://civitai.com/api/v1/" + apiPath +  "/" + id;
    return await request(url);
}

/**
 * Extract file information from the given model version infomation.
 *
 * @param {Object} modelVersionInfo - Model version infomation.
 * @param {(string|null)} [type=null] - Optional select by model type.
 * @param {(string|null)} [fp=null] - Optional select by floating point quantization.
 * @param {(string|null)} [size=null] - Optional select by sizing.
 * @param {(string|null)} [format=null] - Optional select by file format.
 *
 * @returns {Object} - Extracted list of infomation on each file of the given model version.
 */
function civitai_getModelFilesInfo(modelVersionInfo, type = null, fp = null, size = null, format = null) {
    const files = [];
    const modelVersionFiles = modelVersionInfo["files"];
    for (let i = 0; i < modelVersionFiles.length; i++) {
        const modelVersionFile = modelVersionFiles[i];
        
        const fileType = modelVersionFile["type"];
        if (type instanceof String && type != fileType) { continue; }
        
        const fileMeta = modelVersionFile["metadata"];
        
        const fileFp = fileMeta["fp"];
        if (fp instanceof String && fp != fileFp) { continue; }
        
        const fileSize = fileMeta["size"];
        if (size instanceof String && size != fileSize) { continue; }
        
        const fileFormat = fileMeta["format"];
        if (format instanceof String && format != fileFormat) { continue; }
        
        files.push({
            "downloadUrl": modelVersionFile["downloadUrl"],
            "format": fileFormat,
            "fp": fileFp,
            "hashes": modelVersionFile["hashes"],
            "name": modelVersionFile["name"],
            "size": fileSize,
            "sizeKB": modelVersionFile["sizeKB"],
            "type": fileType,
        });
    }
    return {
        "files": files,
        "id": modelVersionInfo["id"],
        "images": modelVersionInfo["images"].map((image) => {
            // TODO: do I need to double-check image matches resource?
            return image["url"];
        }),
        "name": modelVersionInfo["name"],
    };
}

/**
 * 
 *
 * @param {string} stringUrl - Model url.
 *
 * @returns {Object} - Download information for the given url.
 */
async function civitai_getFilteredInfo(stringUrl) {
    const url = new URL(stringUrl);
    if (url.hostname != "civitai.com") { return {}; }
    if (url.pathname == "/") { return {} }
    const urlPath = url.pathname;
    if (urlPath.startsWith("/api")) {
        const idEnd = urlPath.length - (urlPath.at(-1) == "/" ? 1 : 0);
        const idStart = urlPath.lastIndexOf("/", idEnd - 1) + 1;
        const modelVersionId = urlPath.substring(idStart, idEnd);
        if (parseInt(modelVersionId, 10) == NaN) {
            return {};
        }
        const modelVersionInfo = await civitai_requestInfo(modelVersionId, "model-versions");
        if (Object.keys(modelVersionInfo).length == 0) {
            return {};
        }
        const searchParams = url.searchParams;
        const filesInfo = civitai_getModelFilesInfo(
            modelVersionInfo,
            searchParams.get("type"),
            searchParams.get("fp"),
            searchParams.get("size"),
            searchParams.get("format"),
        );
        return {
            "name": modelVersionInfo["model"]["name"],
            "type": modelVersionInfo["model"]["type"],
            "versions": [filesInfo]
        }
    }
    else if (urlPath.startsWith('/models')) {
        const idStart = urlPath.indexOf("models/") + "models/".length;
        const idEnd = (() => {
            const idEnd = urlPath.indexOf("/", idStart);
            return idEnd === -1 ? urlPath.length : idEnd;
        })();
        const modelId = urlPath.substring(idStart, idEnd);
        if (parseInt(modelId, 10) == NaN) {
            return {};
        }
        const modelInfo = await civitai_requestInfo(modelId, "models");
        if (Object.keys(modelInfo).length == 0) {
            return {};
        }
        const modelVersionId = parseInt(url.searchParams.get("modelVersionId"));
        const modelVersions = [];
        const modelVersionInfos = modelInfo["modelVersions"];
        for (let i = 0; i < modelVersionInfos.length; i++) {
            const versionInfo = modelVersionInfos[i];
            if (!Number.isNaN(modelVersionId)) {
                if (modelVersionId != versionInfo["id"]) {continue; }
            }
            const filesInfo = civitai_getModelFilesInfo(versionInfo);
            modelVersions.push(filesInfo);
        }
        return {
            "name": modelInfo["name"],
            "type": modelInfo["type"],
            "versions": modelVersions
        }
    }
    else {
        return {};
    }
}

/**
 * Get model info from Huggingface.
 *
 * @param {string} id - Model ID.
 * @param {string} apiPath - API path.
 *
 * @returns {Promise<Object>} Dictionary containing recieved model info. Returns an empty if fails.
 */
async function huggingFace_requestInfo(id, apiPath = "models") {
    const url = "https://huggingface.co/api/" + apiPath + "/" + id;
    return await request(url);
}

/**
 * 
 *
 * @param {string} stringUrl - Model url.
 *
 * @returns {Promise<Object>}
 */
async function huggingFace_getFilteredInfo(stringUrl) {
    const url = new URL(stringUrl);
    if (url.hostname != "huggingface.co") { return {}; }
    if (url.pathname == "/") { return {} }
    const urlPath = url.pathname;
    const i0 = 1;
    const i1 = urlPath.indexOf("/", i0);
    if (i1 == -1 || urlPath.length - 1 == i1) {
        // user-name only
        return {};
    }
    let i2 = urlPath.indexOf("/", i1 + 1);
    if (i2 == -1) {
        // model id only
        i2 = urlPath.length;
    }
    const modelId = urlPath.substring(i0, i2);
    const urlPathEnd = urlPath.substring(i2);
    
    const isValidBranch = (
        urlPathEnd.startsWith("/resolve") ||
        urlPathEnd.startsWith("/blob") ||
        urlPathEnd.startsWith("/tree")
    );
    
    let branch = "/main";
    let filePath = "";
    if (isValidBranch) {
        const i0 = branch.length;
        const i1 = urlPathEnd.indexOf("/", i0 + 1);
        if (i1 == -1) {
            if (i0 != urlPathEnd.length) {
                // ends with branch
                branch = urlPathEnd.substring(i0);
            }
        }
        else {
            branch = urlPathEnd.substring(i0, i1);
            if (urlPathEnd.length - 1 > i1) {
                filePath = urlPathEnd.substring(i1);
            }
        }
    }
    
    const modelInfo = await huggingFace_requestInfo(modelId);
    //const modelInfo = await requestInfo(modelId + "/tree" + branch); // this only gives you the files at the given branch path...
    // oid: SHA-1?, lfs.oid: SHA-256

    const clippedFilePath = filePath.substring(filePath[0] === "/" ? 1 : 0);
    const modelFiles = modelInfo["siblings"].filter((sib) => {
        const filename = sib["rfilename"];
        for (let i = 0; i < MODEL_EXTENSIONS.length; i++) {
            if (filename.endsWith(MODEL_EXTENSIONS[i])) {
                return filename.startsWith(clippedFilePath);
            }
        }
        return false;
    }).map((sib) => {
        const filename = sib["rfilename"];
        return filename;
    });
    if (modelFiles.length === 0) {
        return {};
    }
    
    const baseDownloadUrl = url.origin + urlPath.substring(0, i2) + "/resolve" + branch;
    
    const images = modelInfo["siblings"].filter((sib) => {
        const filename = sib["rfilename"];
        for (let i = 0; i < IMAGE_EXTENSIONS.length; i++) {
            if (filename.endsWith(IMAGE_EXTENSIONS[i])) {
                return filename.startsWith(clippedFilePath);
            }
        }
        return false;
    }).map((sib) => {
        return baseDownloadUrl + "/" + sib["rfilename"];
    });
    
    return {
        "baseDownloadUrl": baseDownloadUrl,
        "modelFiles": modelFiles,
        "images": images,
    };
}

/**
 * @typedef {Object} DirectoryItem
 * @param {string} name
 * @param {number | undefined} childCount
 * @param {number | undefined} childIndex
 */

class DirectoryDropdown {
    /** @type {HTMLDivElement} */
    element = undefined;
    
    /** @type {Boolean} */
    showDirectoriesOnly = false;

    /** @type {HTMLInputElement} */
    #input = undefined;
    
    // TODO: remove this
    /** @type {Function} */
    #updateDropdown = null;
    
    /** @type {Function} */
    #updateCallback = null;
    
    /** @type {Function} */
    #submitCallback = null;
    
    /**
     * @param {HTMLInputElement} input
     * @param {Function} updateDropdown
     * @param {Function} [updateCallback= () => {}]
     * @param {Function} [submitCallback= () => {}]
     * @param {String} [searchSeparator="/"]
     * @param {Boolean} [showDirectoriesOnly=false]
     */
    constructor(input, updateDropdown, updateCallback = () => {}, submitCallback = () => {}, searchSeparator = "/", showDirectoriesOnly = false) {
        /** @type {HTMLDivElement} */
        const dropdown = $el("div.search-dropdown", { // TODO: change to `search-directory-dropdown`
            style: {
                display: "none",
            },
        });
        this.element = dropdown;
        this.#input = input;
        this.#updateDropdown = updateDropdown;
        this.#updateCallback = updateCallback;
        this.#submitCallback = submitCallback;
        this.showDirectoriesOnly = showDirectoriesOnly;
        
        input.addEventListener("input", () => updateDropdown());
        input.addEventListener("focus", () => updateDropdown());
        input.addEventListener("blur", () => { dropdown.style.display = "none"; });
        input.addEventListener(
            "keydown",
            (e) => {
                const options = dropdown.children;
                let iSelection;
                for (iSelection = 0; iSelection < options.length; iSelection++) {
                    const selection = options[iSelection];
                    if (selection.classList.contains(DROPDOWN_DIRECTORY_SELECTION_CLASS)) {
                        break;
                    }
                }
                if (e.key === "Escape") {
                    e.stopPropagation();
                    if (iSelection < options.length) {
                        const selection = options[iSelection];
                        selection.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                    }
                    else {
                        e.target.blur();
                    }
                }
                else if (e.key === "ArrowRight" && dropdown.style.display !== "none") {
                    const selection = options[iSelection];
                    if (selection !== undefined && selection !== null) {
                        e.stopPropagation();
                        e.preventDefault(); // prevent cursor move
                        const input = e.target;
                        DirectoryDropdown.selectionToInput(input, selection, searchSeparator);
                        updateDropdown();
                        //updateCallback();
                        //submitCallback();
                        /*
                        const options = dropdown.children;
                        if (options.length > 0) {
                            // arrow key navigation
                            options[0].classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                        }
                        */
                    }
                }
                else if (e.key === "ArrowLeft" && dropdown.style.display !== "none") {
                    const input = e.target;
                    const oldFilterText = input.value;
                    const iSep = oldFilterText.lastIndexOf(searchSeparator, oldFilterText.length - 2);
                    const newFilterText = oldFilterText.substring(0, iSep + 1);
                    if (oldFilterText !== newFilterText) {
                        const delta = oldFilterText.substring(iSep + 1);
                        let isMatch = delta[delta.length-1] === searchSeparator;
                        if (!isMatch) {
                            const options = dropdown.children;
                            for (let i = 0; i < options.length; i++) {
                                const option = options[i];
                                if (option.innerText.startsWith(delta)) {
                                    isMatch = true;
                                    break;
                                }
                            }
                        }
                        if (isMatch) {
                            e.stopPropagation();
                            e.preventDefault(); // prevent cursor move
                            input.value = newFilterText;
                            updateDropdown();
                            //updateCallback();
                            //submitCallback();
                            /*
                            const options = dropdown.children;
                            let isSelected = false;
                            for (let i = 0; i < options.length; i++) {
                                const option = options[i];
                                if (option.innerText.startsWith(delta)) {
                                    option.classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                                    isSelected = true;
                                    break;
                                }
                            }
                            if (!isSelected) {
                                const options = dropdown.children;
                                if (options.length > 0) {
                                    // arrow key navigation
                                    options[0].classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                                }
                            }
                            */
                        }
                    }
                }
                else if (e.key === "Enter") {
                    e.stopPropagation();
                    const input = e.target
                    const selection = options[iSelection];
                    if (selection !== undefined && selection !== null) {
                        DirectoryDropdown.selectionToInput(input, selection, searchSeparator);
                        updateDropdown();
                        updateCallback();
                    }
                    submitCallback();
                    input.blur();
                }
                else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && dropdown.style.display !== "none") {
                    e.stopPropagation();
                    e.preventDefault(); // prevent cursor move
                    let iNext = options.length;
                    if (iSelection < options.length) {
                        const selection = options[iSelection];
                        selection.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                        const delta = e.key === "ArrowDown" ? 1 : -1;
                        iNext = iSelection + delta;
                        if (0 <= iNext && iNext < options.length) {
                            const selectionNext = options[iNext];
                            selectionNext.classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                        }
                    }
                    else if (iSelection === options.length) {
                        iNext = e.key === "ArrowDown" ? 0 : options.length-1;
                        const selection = options[iNext]
                        selection.classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                    }
                    if (0 <= iNext && iNext < options.length) {
                        let dropdownTop = dropdown.scrollTop;
                        const dropdownHeight = dropdown.offsetHeight;
                        const selection = options[iNext];
                        const selectionHeight = selection.offsetHeight;
                        const selectionTop = selection.offsetTop;
                        dropdownTop = Math.max(dropdownTop, selectionTop - dropdownHeight + selectionHeight);
                        dropdownTop = Math.min(dropdownTop, selectionTop);
                        dropdown.scrollTop = dropdownTop;
                    }
                    else {
                        dropdown.scrollTop = 0;
                        const options = dropdown.children;
                        for (iSelection = 0; iSelection < options.length; iSelection++) {
                            const selection = options[iSelection];
                            if (selection.classList.contains(DROPDOWN_DIRECTORY_SELECTION_CLASS)) {
                                selection.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                            }
                        }
                    }
                }
            },
        );
    }
    
    /**
     * @param {HTMLInputElement} input
     * @param {HTMLParagraphElement | undefined | null} selection
     * @param {String} searchSeparator
     */
    static selectionToInput(input, selection, searchSeparator) {
        selection.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
        const selectedText = selection.innerText;
        const oldFilterText = input.value;
        const iSep = oldFilterText.lastIndexOf(searchSeparator);
        const previousPath = oldFilterText.substring(0, iSep + 1);
        input.value = previousPath + selectedText;
    }
    
    /**
     * @param {DirectoryItem[]} directories
     * @param {string} searchSeparator
     * @param {string} [modelType = ""]
     */
    update(directories, searchSeparator, modelType = "") {
        const dropdown = this.element;
        const input = this.#input;
        const updateDropdown = this.#updateDropdown;
        const updateCallback = this.#updateCallback;
        const submitCallback = this.#submitCallback;
        const showDirectoriesOnly = this.showDirectoriesOnly;

        const filter = input.value;
        if (filter[0] !== searchSeparator) {
            dropdown.style.display = "none";
            return;
        }

        let cwd = 0;
        if (modelType !== "") {
            const root = directories[0];
            const rootChildIndex = root["childIndex"];
            const rootChildCount = root["childCount"];
            cwd = null;
            for (let i = rootChildIndex; i < rootChildIndex + rootChildCount; i++) {
                const modelDir = directories[i];
                if (modelDir["name"] === modelType) {
                    cwd = i;
                    break;
                }
            }
        }

        // TODO: directories === undefined?
        let indexLastWord = 1;
        while (true) {
            const indexNextWord = filter.indexOf(searchSeparator, indexLastWord);
            if (indexNextWord === -1) {
                // end of filter
                break;
            }

            const item = directories[cwd];
            const childCount = item["childCount"];
            if (childCount === undefined) {
                // file
                break;
            }
            if (childCount === 0) {
                // directory is empty
                break;
            }
            const childIndex = item["childIndex"];
            const items = directories.slice(childIndex, childIndex + childCount);

            const word = filter.substring(indexLastWord, indexNextWord);
            cwd = null;
            for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
                const itemName = items[itemIndex]["name"];
                if (itemName === word) {
                    // directory exists
                    cwd = childIndex + itemIndex;
                    break;
                }
            }
            if (cwd === null) {
                // directory does not exist
                break;
            }
            indexLastWord = indexNextWord + 1;
        }
        if (cwd === null) {
            dropdown.style.display = "none";
            return;
        }

        let options = [];
        const lastWord = filter.substring(indexLastWord);
        const item = directories[cwd];
        if (item["childIndex"] !== undefined) {
            const childIndex = item["childIndex"];
            const childCount = item["childCount"];
            const items = directories.slice(childIndex, childIndex + childCount);
            for (let i = 0; i < items.length; i++) {
                const child = items[i];
                const grandChildCount = child["childCount"];
                const isDir = grandChildCount !== undefined && grandChildCount !== null;
                const itemName = child["name"];
                if (itemName.startsWith(lastWord) && (!showDirectoriesOnly || (showDirectoriesOnly && isDir))) {
                    options.push(itemName + (isDir ? searchSeparator : ""));
                }
            }
        }
        else if (!showDirectoriesOnly) {
            const filename = item["name"];
            if (filename.startsWith(lastWord)) {
                options.push(filename);
            }
        }
        if (options.length === 0) {
            dropdown.style.display = "none";
            return;
        }

        const selection_select = (e) => {
            const selection = e.target;
            if (e.movementX === 0 && e.movementY === 0) { return; }
            if (!selection.classList.contains(DROPDOWN_DIRECTORY_SELECTION_CLASS)) {
                // assumes only one will ever selected at a time
                e.stopPropagation();
                const children = dropdown.children;
                let iChild;
                for (iChild = 0; iChild < children.length; iChild++) {
                    const child = children[iChild];
                    child.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
                }
                selection.classList.add(DROPDOWN_DIRECTORY_SELECTION_CLASS);
            }
        };
        const selection_deselect = (e) => {
            e.stopPropagation();
            e.target.classList.remove(DROPDOWN_DIRECTORY_SELECTION_CLASS);
        };
        const selection_submit = (e) => {
            e.stopPropagation();
            const selection = e.target;
            DirectoryDropdown.selectionToInput(input, selection, searchSeparator);
            updateDropdown();
            updateCallback();e.target
            submitCallback();
        };
        const innerHtml = options.map((text) => {
            /** @type {HTMLParagraphElement} */
            const p = $el(
                "p",
                {
                    onmouseenter: (e) => selection_select(e),
                    onmousemove: (e) => selection_select(e),
                    onmouseleave: (e) => selection_deselect(e),
                    onmousedown: (e) => selection_submit(e),
                },
                [
                    text
                ]
            );
            return p;
        });
        dropdown.innerHTML = "";
        dropdown.append.apply(dropdown, innerHtml);
        // TODO: handle when dropdown is near the bottom of the window
        const inputRect = input.getBoundingClientRect();
        dropdown.style.minWidth = inputRect.width + "px";
        dropdown.style.top = (input.offsetTop + inputRect.height) + "px";
        dropdown.style.left = input.offsetLeft + "px";
        dropdown.style.display = "block";
    }
}

/**
 * @param {string} nodeType
 * @returns {int}
 */
function modelWidgetIndex(nodeType) {
    return 0;
}

/**
 * @param {string} path
 * @returns {string}
 */
function pathToFileString(path) {
    const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\")) + 1;
    return path.slice(i);
}

/**
 * @param {string} path
 * @returns {string}
 */
function searchPathToSystemPath(path, searchSeparator, systemSeparator) {
    const i1 = path.indexOf(searchSeparator, 1);
    const i2 = path.indexOf(searchSeparator, i1 + 1);
    return path.slice(i2 + 1).replaceAll(searchSeparator, systemSeparator);
}

/**
 * @param {string} file
 * @returns {string | undefined}
 */
function removeModelExtension(file) {
    // This is a bit sloppy (can assume server sends without)
    const i = file.lastIndexOf(".");
    if (i != -1) {
        return file.substring(0, i);
    }
}

/**
 * @param {string} text
 * @param {string} file
 * @param {boolean} removeExtension
 * @returns {string}
 */
function insertEmbeddingIntoText(text, file, removeExtension) {
    let name = file;
    if (removeExtension) {
        name = removeModelExtension(name)
    }
    const sep = text.length === 0 || text.slice(-1).match(/\s/) ? "" : " ";
    return text + sep + "(embedding:" +  name  + ":1.0)";
}

/**
 * @param {HTMLButtonElement} element
 * @param {boolean} success
 * @param {string} [successText=""]
 * @param {string} [failureText=""]
 * @param {string} [resetText=""]
 */
function buttonAlert(element, success, successText = "", failureText = "", resetText = "") {
    if (element === undefined || element === null) {
        return;
    }
    const name = success ? "button-success" : "button-failure";
    element.classList.add(name);
    if (successText != "" && failureText != "") {
        element.innerHTML = success ? successText : failureText;
    }
    // TODO: debounce would be nice to get working...
    window.setTimeout((element, name, innerHTML) => {
        element.classList.remove(name);
        if (innerHTML != "") {
            element.innerHTML = innerHTML;
        }
    }, 1000, element, name, resetText);
}

class Tabs {
    /** @type {Record<string, HTMLDivElement>} */
    #head = {};
    /** @type {Record<string, HTMLDivElement>} */
    #body = {};

    /**
     * @param {HTMLDivElement[]} tabs
     */
    constructor(tabs) {
        const head = [];
        const body = [];

        tabs.forEach((el, index) => {
            const name = el.getAttribute("data-name");

            /** @type {HTMLDivElement} */
            const tag = $el(
                "div.head-item",
                { onclick: () => this.active(name) },
                [name]
            );

            if (index === 0) {
                this.#active = name;
            }

            this.#head[name] = tag;
            head.push(tag);
            this.#body[name] = el;
            body.push(el);
        });

        this.element = $el("div.comfy-tabs", [
            $el("div.comfy-tabs-head", head),
            $el("div.comfy-tabs-body", body),
        ]);

        this.active(this.#active);
    }

    #active = undefined;

    /**
     * @param {string} name
     */
    active(name) {
        this.#active = name;
        Object.keys(this.#head).forEach((key) => {
            if (name === key) {
                this.#head[key].classList.add("active");
                this.#body[key].style.display = "";
            } else {
                this.#head[key].classList.remove("active");
                this.#body[key].style.display = "none";
            }
        });
    }
}

/**
 * @param {Record<HTMLDivElement, Any>} tabs
 * @returns {HTMLDivElement[]}
 */
function $tabs(tabs) {
    const instance = new Tabs(tabs);
    return instance.element;
}

/**
 * @param {string} name
 * @param {HTMLDivElement[]} el
 * @returns {HTMLDivElement}
 */
function $tab(name, el) {
    return $el("div", { dataset: { name } }, el);
}

/**
 * @returns {HTMLLabelElement}
 */
function $checkbox(x = { $: (el) => {}, textContent: "", checked: false }) {
    const text = x.textContent;
    const input = $el("input", {
        type: "checkbox", 
        checked: x.checked ?? false, 
    });
    const label = $el("label", [
        input, 
        text === "" || text === undefined || text === null ? "" : " " + text,
    ]);
    if (x.$ !== undefined){
        x.$(input);
    }
    return label;
}

class ModelGrid {
    /**
     * @param {Array} list
     * @param {string} searchString
     * @returns {Array}
     */
    static #filter(list, searchString) {
        /** @type {string[]} */
        const keywords = searchString
            //.replace("*", " ") // TODO: this is wrong for wildcards
            .split(/(-?".*?"|[^\s"]+)+/g)
            .map((item) => item
                .trim()
                .replace(/(?:")+/g, "")
                .toLowerCase())
            .filter(Boolean);

        const regexSHA256 = /^[a-f0-9]{64}$/gi;
        const fields = ["name", "path"];
        return list.filter((element) => {
            const text = fields
                .reduce((memo, field) => memo + " " + element[field], "")
                .toLowerCase();
            return keywords.reduce((memo, target) => {
                const excludeTarget = target[0] === "-";
                if (excludeTarget && target.length === 1) { return memo; }
                const filteredTarget = excludeTarget ? target.slice(1) : target;
                if (element["SHA256"] !== undefined && regexSHA256.test(filteredTarget)) {
                    return memo && excludeTarget !== (filteredTarget === element["SHA256"]);
                }
                else {
                    return memo && excludeTarget !== text.includes(filteredTarget);
                }
            }, true);
        });
    }
    
    /**
     * In-place sort. Returns an arrat alias.
     * @param {Array} list
     * @param {string} sortBy
     * @param {bool} [reverse=false]
     * @returns {Array}
     */
    static #sort(list, sortBy, reverse = false) {
        let compareFn = undefined;
        switch (sortBy) {
            case MODEL_SORT_DATE_NAME:
                compareFn = (a, b) => { return a[MODEL_SORT_DATE_NAME].localeCompare(b[MODEL_SORT_DATE_NAME]); };
                break;
            case MODEL_SORT_DATE_MODIFIED:
                compareFn = (a, b) => { return b[MODEL_SORT_DATE_MODIFIED] - a[MODEL_SORT_DATE_MODIFIED]; };
                break;
            case MODEL_SORT_DATE_CREATED:
                compareFn = (a, b) => { return b[MODEL_SORT_DATE_CREATED] - a[MODEL_SORT_DATE_CREATED]; };
                break;
            default:
                console.warn("Invalid filter sort value: '" + sortBy + "'");
                return list;
        }
        const sorted = list.sort(compareFn);
        return reverse ? sorted.reverse() : sorted;
    }

    /**
     * @param {Event} event
     * @param {string} modelType
     * @param {string} path
     * @param {boolean} removeEmbeddingExtension
     * @param {int} addOffset
     */
    static #addModel(event, modelType, path, removeEmbeddingExtension, addOffset) {
        let success = false;
        if (modelType !== "embeddings") {
            const nodeType = modelNodeType[modelType];
            const widgetIndex = modelWidgetIndex(nodeType);
            let node = LiteGraph.createNode(nodeType, null, []);
            if (node) {
                node.widgets[widgetIndex].value = path;
                const selectedNodes = app.canvas.selected_nodes;
                let isSelectedNode = false;
                for (var i in selectedNodes) {
                    const selectedNode = selectedNodes[i];
                    node.pos[0] = selectedNode.pos[0] + addOffset;
                    node.pos[1] = selectedNode.pos[1] + addOffset;
                    isSelectedNode = true;
                    break;
                }
                if (!isSelectedNode) {
                    const graphMouse = app.canvas.graph_mouse;
                    node.pos[0] = graphMouse[0];
                    node.pos[1] = graphMouse[1];
                }
                app.graph.add(node, {doProcessChange: true});
                app.canvas.selectNode(node);
                success = true;
            }
            event.stopPropagation();
        }
        else if (modelType === "embeddings") {
            const embeddingFile = pathToFileString(path);
            const selectedNodes = app.canvas.selected_nodes;
            for (var i in selectedNodes) {
                const selectedNode = selectedNodes[i];
                const nodeType = modelNodeType[modelType];
                const widgetIndex = modelWidgetIndex(nodeType);
                const target = selectedNode.widgets[widgetIndex].element;
                if (target && target.type === "textarea") {
                    target.value = insertEmbeddingIntoText(target.value, embeddingFile, removeEmbeddingExtension);
                    success = true;
                }
            }
            if (!success) {
                console.warn("Try selecting a node before adding the embedding.");
            }
            event.stopPropagation();
        }
        buttonAlert(event.target, success, "✔", "✖", "✚");
    }

    /**
     * @param {Event} event
     * @param {string} modelType
     * @param {string} path
     * @param {boolean} removeEmbeddingExtension
     * @param {boolean} strictDragToAdd
     */
    static #dragAddModel(event, modelType, path, removeEmbeddingExtension, strictDragToAdd) {
        const target = document.elementFromPoint(event.x, event.y);
        if (modelType !== "embeddings" && target.id === "graph-canvas") {
            const nodeType = modelNodeType[modelType];
            const widgetIndex = modelWidgetIndex(nodeType);
            const pos = app.canvas.convertEventToCanvasOffset(event);
            const nodeAtPos = app.graph.getNodeOnPos(pos[0], pos[1], app.canvas.visible_nodes);

            let draggedOnNode = nodeAtPos && nodeAtPos.type === nodeType;
            if (strictDragToAdd) {
                const draggedOnWidget = app.canvas.processNodeWidgets(nodeAtPos, pos, event) === nodeAtPos.widgets[widgetIndex];
                draggedOnNode = draggedOnNode && draggedOnWidget;
            }

            if (draggedOnNode) {
                let node = nodeAtPos;
                node.widgets[widgetIndex].value = path;
                app.canvas.selectNode(node);
            }
            else {
                let node = LiteGraph.createNode(nodeType, null, []);
                if (node) {
                    node.pos[0] = pos[0];
                    node.pos[1] = pos[1];
                    node.widgets[widgetIndex].value = path;
                    app.graph.add(node, {doProcessChange: true});
                    app.canvas.selectNode(node);
                }
            }
            event.stopPropagation();
        }
        else if (modelType === "embeddings" && target.type === "textarea") {
            const pos = app.canvas.convertEventToCanvasOffset(event);
            const nodeAtPos = app.graph.getNodeOnPos(pos[0], pos[1], app.canvas.visible_nodes);
            if (nodeAtPos) {
                app.canvas.selectNode(nodeAtPos);
                const embeddingFile = pathToFileString(path);
                target.value = insertEmbeddingIntoText(target.value, embeddingFile, removeEmbeddingExtension);
                event.stopPropagation();
            }
        }
    }

    /**
     * @param {Event} event
     * @param {string} modelType
     * @param {string} path
     * @param {boolean} removeEmbeddingExtension
     */
    static #copyModelToClipboard(event, modelType, path, removeEmbeddingExtension) {
        const nodeType = modelNodeType[modelType];
        let success = false;
        if (nodeType === "Embedding") {
            if (navigator.clipboard){
                const embeddingFile = pathToFileString(path);
                const embeddingText = insertEmbeddingIntoText("", embeddingFile, removeEmbeddingExtension);
                navigator.clipboard.writeText(embeddingText);
                success = true;
            }
            else {
                console.warn("Cannot copy the embedding to the system clipboard; Try dragging it instead.");
            }
        }
        else if (nodeType) {
            const node = LiteGraph.createNode(nodeType, null, []);
            const widgetIndex = modelWidgetIndex(nodeType);
            node.widgets[widgetIndex].value = path;
            app.canvas.copyToClipboard([node]);
            success = true;
        }
        else {
            console.warn(`Unable to copy unknown model type '${modelType}.`);
        }
        buttonAlert(event.target, success, "✔", "✖", "⧉︎");
    }

    /**
     * @param {Array} models
     * @param {string} modelType
     * @param {Object.<HTMLInputElement>} settingsElements
     * @param {String} searchSeparator
     * @param {String} systemSeparator
     * @param {Function} modelInfoCallback
     * @returns {HTMLElement[]}
     */
    static #generateInnerHtml(models, modelType, settingsElements, searchSeparator, systemSeparator, modelInfoCallback) {
        // TODO: seperate text and model logic; getting too messy
        // TODO: fallback on button failure to copy text?
        const canShowButtons = modelNodeType[modelType] !== undefined;
        const showAddButton = canShowButtons && settingsElements["model-show-add-button"].checked;
        const showCopyButton = canShowButtons && settingsElements["model-show-copy-button"].checked;
        const strictDragToAdd = settingsElements["model-add-drag-strict-on-field"].checked;
        const addOffset = parseInt(settingsElements["model-add-offset"].value);
        const showModelExtension = settingsElements["model-show-label-extensions"].checked;
        const removeEmbeddingExtension = !settingsElements["model-add-embedding-extension"].checked;
        if (models.length > 0) {
            return models.map((item) => {
                const previewInfo = item.preview;
                const searchPath = item.path;
                const path = searchPathToSystemPath(searchPath, searchSeparator, systemSeparator);
                let buttons = [];
                if (showAddButton) {
                    buttons.push(
                        $el("button.icon-button.model-button", {
                            type: "button",
                            textContent: "⧉︎",
                            onclick: (e) => ModelGrid.#copyModelToClipboard(
                                e, 
                                modelType, 
                                path, 
                                removeEmbeddingExtension
                            ),
                            draggable: false,
                        })
                    );
                }
                if (showCopyButton) {
                    buttons.push(
                        $el("button.icon-button.model-button", {
                            type: "button",
                            textContent: "✚",
                            onclick: (e) => ModelGrid.#addModel(
                                e, 
                                modelType, 
                                path, 
                                removeEmbeddingExtension, 
                                addOffset
                            ),
                            draggable: false,
                        })
                    );
                }
                const dragAdd = (e) => ModelGrid.#dragAddModel(
                    e, 
                    modelType, 
                    path, 
                    removeEmbeddingExtension, 
                    strictDragToAdd
                );
                return $el("div.item", {}, [
                    $el("img.model-preview", {
                        src: imageUri(previewInfo?.path, previewInfo?.dateModified),
                        draggable: false,
                    }),
                    $el("div.model-preview-overlay", {
                        ondragend: (e) => dragAdd(e),
                        draggable: true,
                    }),
                    $el("div.model-preview-top-right", {
                        draggable: false,
                    },
                        buttons
                    ),
                    $el("div.model-preview-top-left", {
                        draggable: false,
                    }, [
                        $el("button.icon-button.model-button", {
                            type: "button",
                            textContent: "ⓘ",
                            onclick: async() => modelInfoCallback(searchPath),
                            draggable: false,
                        }),
                    ]),
                    $el("div.model-label", {
                        ondragend: (e) => dragAdd(e),
                        draggable: true,
                    }, [
                        $el("p", [showModelExtension ? item.name : removeModelExtension(item.name)])
                    ]),
                ]);
            });
        } else {
            return [$el("h2", ["No Models"])];
        }
    }

    /**
     * @param {HTMLDivElement} modelGrid
     * @param {Object} models
     * @param {HTMLSelectElement} modelSelect
     * @param {Object.<{value: string}>} previousModelType
     * @param {Object} settings
     * @param {string} sortBy
     * @param {boolean} reverseSort
     * @param {Array} previousModelFilters
     * @param {HTMLInputElement} modelFilter
     * @param {String} searchSeparator
     * @param {String} systemSeparator
     * @param {Function} modelInfoCallback
     */
    static update(modelGrid, models, modelSelect, previousModelType, settings, sortBy, reverseSort, previousModelFilters, modelFilter, searchSeparator, systemSeparator, modelInfoCallback) {
        let modelType = modelSelect.value;
        if (models[modelType] === undefined) {
            modelType = "checkpoints"; // TODO: magic value
        }

        if (modelType !== previousModelType.value) {
            if (settings["model-persistent-search"].checked) {
                previousModelFilters.splice(0, previousModelFilters.length); // TODO: make sure this actually worked!
            }
            else {
                // cache previous filter text
                previousModelFilters[previousModelType.value] = modelFilter.value;
                // read cached filter text
                modelFilter.value = previousModelFilters[modelType] ?? "";
            }
            previousModelType.value = modelType;
        }

        let modelTypeOptions = [];
        for (const [key, value] of Object.entries(models)) {
            const el = $el("option", [key]);
            modelTypeOptions.push(el);
        }
        modelSelect.innerHTML = "";
        modelTypeOptions.forEach(option => modelSelect.add(option));
        modelSelect.value = modelType;

        const searchAppend = settings["model-search-always-append"].value;
        const searchText = modelFilter.value + " " + searchAppend;
        const modelList = ModelGrid.#filter(models[modelType], searchText);
        ModelGrid.#sort(modelList, sortBy, reverseSort);

        modelGrid.innerHTML = "";
        const modelGridModels = ModelGrid.#generateInnerHtml(
            modelList, 
            modelType, 
            settings, 
            searchSeparator, 
            systemSeparator,
            modelInfoCallback,
        );
        modelGrid.append.apply(modelGrid, modelGridModels);
    }
}

/**
 * @param {Any} attr
 * @returns {HTMLDivElement}
 */
function $radioGroup(attr) {
    const { name = Date.now(), onchange, options = [], $ } = attr;

    /** @type {HTMLDivElement[]} */
    const radioGroup = options.map((item, index) => {
        const inputRef = { value: null };

        return $el(
            "div.comfy-radio",
            { onclick: () => inputRef.value.click() },
            [
                $el("input.radio-input", {
                    type: "radio",
                    name: name,
                    value: item.value,
                    checked: index === 0,
                    $: (el) => (inputRef.value = el),
                }),
                $el("label", [item.label ?? item.value]),
            ]
        );
    });

    const element = $el("input", { value: options[0]?.value });
    $?.(element);

    radioGroup.forEach((radio) => {
        radio.addEventListener("change", (event) => {
            const selectedValue = event.target.value;
            element.value = selectedValue;
            onchange?.(selectedValue);
        });
    });

    return $el("div.comfy-radio-group", radioGroup);
}

/**
 * @param {HTMLDivElement} previewImageContainer
 * @param {Event} e 
 * @param {1 | -1} step 
 */
function updateRadioPreview(previewImageContainer, step) {
    const children = previewImageContainer.children;
    if (children.length === 0) {
        return;
    }
    let currentIndex = -step;
    for (let i = 0; i < children.length; i++) {
        const previewImage = children[i];
        const display = previewImage.style.display;
        if (display !== "none") {
            currentIndex = i;
        }
        previewImage.style.display = "none";
    }
    currentIndex = currentIndex + step;
    if (currentIndex >= children.length) { currentIndex = 0; }
    else if (currentIndex < 0) { currentIndex = children.length - 1; }
    children[currentIndex].style.display = "block";
}

/**
 * @param {String} uniqueName
 * @param {String[]} defaultPreviews
 * @returns {[]}
 */
function radioGroupImageSelect(uniqueName, defaultPreviews, defaultChanges=false) {
    const defaultImageCount = defaultPreviews.length;
    
    const el_defaultUri = $el("div", {
        style: { display: "none" },
        "data-noimage": imageUri(),
    });
    
    const el_noImage = $el("img", {
        src: imageUri(),
        style: {
            display: defaultImageCount === 0 ? "block" : "none",
        },
        loading: "lazy",
    });
    
    const el_defaultImages = $el("div", {
        style: {
            width: "100%",
            height: "100%",
        },
    }, (() => {
        const imgs = defaultPreviews.map((url) => {
            return $el("img", {
                src: url,
                style: { display: "none" },
                loading: "lazy",
                onerror: (e) => {
                    e.target.src = el_defaultUri.dataset.noimage ?? imageUri();
                },
            });
        });
        if (imgs.length > 0) {
            imgs[0].style.display = "block";
        }
        return imgs;
    })());
    
    const el_uploadImage = $el("img", {
        src: imageUri(),
        style: { display : "none" },
        onerror: (e) => {
            e.target.src = el_defaultUri.dataset.noimage ?? imageUri();
        },
    });
    const el_uploadFile = $el("input", {
        type: "file",
        accept: IMAGE_EXTENSIONS.join(", "),
        onchange: (e) => {
            const file = e.target.files[0];
            if (file) {
                el_uploadImage.src = URL.createObjectURL(file);
            }
            else {
                el_uploadImage.src = el_defaultUri.dataset.noimage;
            }
        },
    });
    const el_upload = $el("div", {
        style: { display: "none" },
    }, [
        el_uploadFile,
    ]);
    
    const el_urlImage = $el("img", {
        src: imageUri(),
        style: { display: "none" },
        onerror: (e) => {
            e.target.src = el_defaultUri.dataset.noimage ?? imageUri();
        },
    });
    const el_customUrl = $el("input.search-text-area", {
        type: "text",
        placeholder: "https://custom-image-preview.png",
    });
    const el_custom = $el("div.row.tab-header-flex-block", {
        style: { display: "none" },
    }, [
        el_customUrl,
        $el("button.icon-button", {
            textContent: "🔍︎",
            onclick: (e) => {
                el_urlImage.src = el_customUrl.value;
            },
        }),
    ]);
    
    const el_previewButtons = $el("div.model-preview-overlay", {
        style: {
            display: el_defaultImages.children.length > 1 ? "block" : "none",
        },
    }, [
        $el("button.icon-button.model-preview-button-left", {
            textContent: "←",
            onclick: () => updateRadioPreview(el_defaultImages, -1),
        }),
        $el("button.icon-button.model-preview-button-right", {
            textContent: "→",
            onclick: () => updateRadioPreview(el_defaultImages, 1),
        }),
    ]);
    const previews = [
        el_noImage,
        el_defaultImages,
        el_urlImage,
        el_uploadImage,
    ];
    const el_preview = $el("div.item", [
        $el("div", {
                style: {
                    "width": "100%",
                    "height": "100%",
                },
            }, 
            previews,
        ),
        el_previewButtons,
    ]);
    
    const PREVIEW_NONE = "No Preview";
    const PREVIEW_DEFAULT = "Default";
    const PREVIEW_URL = "URL";
    const PREVIEW_UPLOAD = "Upload";
    
    const el_radioButtons = $radioGroup({
        name: uniqueName,
        onchange: (value) => {
            el_custom.style.display = "none";
            el_upload.style.display = "none";
            
            el_defaultImages.style.display = "none";
            el_previewButtons.style.display = "none";
            
            el_noImage.style.display = "none";
            el_uploadImage.style.display = "none";
            el_urlImage.style.display = "none";
            
            switch (value) {
                case PREVIEW_NONE:
                default:
                    el_noImage.style.display = "block";
                    break;
                case PREVIEW_DEFAULT:
                    el_defaultImages.style.display = "block";
                    el_previewButtons.style.display = el_defaultImages.children.length > 1 ? "block" : "none";
                    break;
                case PREVIEW_URL:
                    el_custom.style.display = "flex";
                    el_urlImage.style.display = "block";
                    break;
                case PREVIEW_UPLOAD:
                    el_upload.style.display = "flex";
                    el_uploadImage.style.display = "block";
                    break;
            }
        },
        options: (() => {
            const radios = [];
            radios.push({ value: PREVIEW_NONE });
            if (defaultImageCount > 0) {
                radios.push({ value: PREVIEW_DEFAULT });
            }
            radios.push({ value: PREVIEW_URL });
            radios.push({ value: PREVIEW_UPLOAD })
            return radios;
        })(),
    });
    
    if (defaultImageCount > 0) {
        const children = el_radioButtons.children;
        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            const radioButton = child.children[0];
            if (radioButton.value === PREVIEW_DEFAULT) {
                radioButton.checked = true;
                break;
            }
        };
    }
    
    const resetModelInfoPreview = () => {
        let noimage = el_defaultUri.dataset.noimage;
        previews.forEach((el) => {
            el.style.display = "none";
            if (el_noImage !== el) {
                if (el.nodeName === "IMG") {
                    el.src = noimage;
                }
                else {
                    el.children[0].src = noimage;
                }
            }
            else {
                el.src = imageUri();
            }
        });
        const children = el_radioButtons.children;
        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            const radioButton = child.children[0];
            if (radioButton.value === PREVIEW_DEFAULT) {
                el_defaultImages.style.display = "block";
                radioButton.checked = true;
                break;
            }
        };
        el_uploadFile.value = "";
        el_customUrl.value = "";
        el_upload.style.display = "none";
        el_custom.style.display = "none";
    };
    
    const getImage = () => {
        const value = document.querySelector(`input[name="${uniqueName}"]:checked`).value;
        switch (value) {
            case PREVIEW_DEFAULT:
                if (defaultImageCount === 0) {
                    return "";
                }
                const children = el_defaultImages.children;
                for (let i = 0; i < children.length; i++) {
                    const child = children[i];
                    if (child.style.display !== "none") {
                        return child.src;
                    }
                }
                return "";
            case PREVIEW_URL:
                return el_customUrl.value;
            case PREVIEW_UPLOAD:
                return el_uploadFile.files[0] ?? "";
            case PREVIEW_NONE:
                return imageUri();
        }
        return "";
    };
    
    const el_radioGroup = $el("div.model-preview-select-radio-container", [
        $el("div.row.tab-header-flex-block", [el_radioButtons]),
        $el("div", [
            el_custom,
            el_upload,
        ]),
    ]);
    
    return [el_radioGroup, el_preview, getImage, el_defaultUri, resetModelInfoPreview];
}

class ModelManager extends ComfyDialog {
    #el = {
        /** @type {HTMLDivElement} */ modelInfoView: null,
        /** @type {HTMLDivElement} */ modelInfoContainer: null,
        /** @type {HTMLDivElement} */ modelInfoUrl: null,
        /** @type {HTMLDivElement} */ modelInfoOverwrite: null,
        /** @type {HTMLDivElement} */ modelInfos: null,
        modelInfoPreview: null,
        modelInfoDefaultUri: null,

        /** @type {HTMLDivElement} */ modelGrid: null,
        /** @type {HTMLSelectElement} */ modelTypeSelect: null,
        /** @type {HTMLSelectElement} */ modelSortSelect: null,
        /** @type {HTMLDivElement} */ //searchDirectoryDropdown: null,
        /** @type {HTMLInputElement} */ modelContentFilter: null,

        /** @type {HTMLDivElement} */ sidebarButtons: null,

        /** @type {HTMLDivElement} */ settingsTab: null,
        /** @type {HTMLButtonElement} */ settings_reloadBtn: null,
        /** @type {HTMLButtonElement} */ settings_saveBtn: null,
        settings: {
            //"sidebar-default-height": null,
            //"sidebar-default-width": null,
            /** @type {HTMLTextAreaElement} */ "model-search-always-append": null,
            /** @type {HTMLInputElement} */ "model-persistent-search": null,
            /** @type {HTMLInputElement} */ "model-show-label-extensions": null,
            /** @type {HTMLInputElement} */ "model-show-add-button": null,
            /** @type {HTMLInputElement} */ "model-show-copy-button": null,
            /** @type {HTMLInputElement} */ "model-add-embedding-extension": null,
            /** @type {HTMLInputElement} */ "model-add-drag-strict-on-field": null,
            /** @type {HTMLInputElement} */ "model-add-offset": null,
        }
    };

    #data = {
        /** @type {Object} */ models: {},
        /** @type {DirectoryItem[]} */ modelDirectories: [],
        /** @type {Array} */ previousModelFilters: [],
        /** @type {Object.<{value: string}>} */ previousModelType: { value: undefined },
    };

    /** @type {string} */
    #searchSeparator = "/";

    /** @type {string} */
    #systemSeparator = null;

    #resetModelInfoPreview = () => {};

    constructor() {
        super();

        const moveDestination = $el("input.search-text-area", {
            placeholder: "/",
        });
        let searchDropdown = null;
        searchDropdown = new DirectoryDropdown(
            moveDestination,
            () => {
                searchDropdown.update(
                    this.#data.modelDirectories,
                    this.#searchSeparator,
                );
            },
            () => {},
            () => {},
            this.#searchSeparator,
            true,
        );
        
        const [el_radioGroup, el_preview, getImage, el_defaultUri, resetModelInfoPreview] = radioGroupImageSelect(
            "model-info-preview-model-FYUIKMNVB", 
            [imageUri()],
        );
        el_preview.style.display = "flex";
        this.#el.modelInfoRadioGroup = el_radioGroup;
        this.#el.modelInfoPreview = el_preview;
        this.#el.modelInfoDefaultUri = el_defaultUri;
        this.#resetModelInfoPreview = resetModelInfoPreview;
        
        this.element = $el(
            "div.comfy-modal.model-manager",
            {
                parent: document.body,
            },
            [
                $el("div.comfy-modal-content", [ // TODO: settings.top_bar_left_to_right or settings.top_bar_right_to_left
                    $el("div.model-info-view", {
                        $: (el) => (this.#el.modelInfoView = el),
                        style: { display: "none" },
                    }, [
                        $el("div", {
                            style: {
                                display: "flex",
                                gap: "8px",
                            },
                        }, [
                            $el("button.icon-button", {
                                textContent: "🗑︎",
                                onclick: async(e) => {
                                    const affirmation = "delete";
                                    const confirmation = window.prompt("Type \"" + affirmation + "\" to delete the model PERMANENTLY.\n\nThis includes all image or text files.");
                                    let deleted = false;
                                    if (confirmation === affirmation) {
                                        const container = this.#el.modelInfoContainer;
                                        const path = encodeURIComponent(container.dataset.path);
                                        deleted = await request(
                                            `/model-manager/model/delete?path=${path}`,
                                            {
                                                method: "POST",
                                            }
                                        )
                                        .then((result) => {
                                            const deleted = result["success"];
                                            if (deleted) 
                                            {
                                                container.innerHTML = "";
                                                this.#el.modelInfoView.style.display = "none";
                                                this.#modelTab_updateModels();
                                            }
                                            return deleted;
                                        })
                                        .catch((err) => {
                                            return false;
                                        });
                                    }
                                    if (!deleted) {
                                        buttonAlert(e.target, false);
                                    }
                                },
                            }),
                        ]),
                        $el("div.row.tab-header", {
                            display: "block",
                        }, [
                            $el("div.row.tab-header-flex-block", [
                                $el("div.search-models", [
                                    moveDestination,
                                    searchDropdown.element,
                                ]),
                                $el("button", {
                                    textContent: "Move",
                                    onclick: async(e) => {
                                        const container = this.#el.modelInfoContainer;
                                        const moved = await request(
                                            `/model-manager/model/move`,
                                            {
                                                method: "POST",
                                                body: JSON.stringify({
                                                    "oldFile": container.dataset.path,
                                                    "newDirectory": moveDestination.value,
                                                }),
                                            }
                                        )
                                        .then((result) => {
                                            const moved = result["success"];
                                            if (moved) 
                                            {
                                                container.innerHTML = "";
                                                this.#el.modelInfoView.style.display = "none";
                                                this.#modelTab_updateModels();
                                            }
                                            return moved;
                                        })
                                        .catch(err => {
                                            return false;
                                        });
                                        if (!moved) {
                                            buttonAlert(e.target, false);
                                        }
                                    },
                                }),
                            ]),
                        ]),
                        $el("div.model-info-container", {
                            $: (el) => (this.#el.modelInfoContainer = el),
                            "data-path": "",
                        }),
                    ]),
                    $el("div.topbar-buttons",
                        [
                            $el("div.sidebar-buttons",
                            {
                                $: (el) => (this.#el.sidebarButtons = el),
                            },
                            [
                                $el("button.icon-button", {
                                    textContent: "◧",
                                    onclick: (event) => this.#setSidebar(event),
                                }),
                                $el("button.icon-button", {
                                    textContent: "⬒",
                                    onclick: (event) => this.#setSidebar(event),
                                }),
                                $el("button.icon-button", {
                                    textContent: "⬓",
                                    onclick: (event) => this.#setSidebar(event),
                                }),
                                $el("button.icon-button", {
                                    textContent: "◨",
                                    onclick: (event) => this.#setSidebar(event),
                                }),
                            ]),
                            $el("button.icon-button", {
                                textContent: "✖",
                                onclick: () => {
                                    const infoView = this.#el.modelInfoView;
                                    if (infoView.style.display === "none") {
                                        this.close();
                                    }
                                    else {
                                        infoView.style.display = "none";
                                    }
                                },
                            }),
                        ]
                    ),
                    $tabs([
                        $tab("Download", [this.#downloadTab_new()]),
                        $tab("Models", this.#modelTab_new()),
                        $tab("Settings", [this.#settingsTab_new()]),
                    ]),
                ]),
            ]
        );

        this.#init();
    }

    #init() {
        this.#settingsTab_reload(false);
        this.#modelTab_updateModels();
    }

    /** @type {DirectoryDropdown} */
    #modelContentFilterDirectoryDropdown = null;

    /**
     * @returns {HTMLElement[]}
     */
    #modelTab_new() {
        /** @type {HTMLDivElement} */
        const modelGrid = $el("div.comfy-grid");
        this.#el.modelGrid = modelGrid;

        const searchInput = $el("input.search-text-area", {
            $: (el) => (this.#el.modelContentFilter = el),
            placeholder: "example: /0/1.5/styles/clothing -.pt",
        });

        const searchDropdown = new DirectoryDropdown(
            searchInput,
            this.#modelTab_updateDirectoryDropdown,
            this.#modelTab_updatePreviousModelFilter,
            this.#modelTab_updateModelGrid,
            this.#searchSeparator,
            false,
        );
        this.#modelContentFilterDirectoryDropdown = searchDropdown;

        return [
            $el("div.row.tab-header", [
                $el("div.row.tab-header-flex-block", [
                    $el("button.icon-button", {
                        type: "button",
                        textContent: "⟳",
                        onclick: () => this.#modelTab_updateModels(),
                    }),
                    $el("select.model-select-dropdown", {
                        $: (el) => (this.#el.modelTypeSelect = el),
                        name: "model-type",
                        onchange: () => this.#modelTab_updateModelGrid(),
                    }),
                    $el("select.model-select-dropdown",
                        {
                            $: (el) => (this.#el.modelSortSelect = el),
                            onchange: () => this.#modelTab_updateModelGrid(),
                        },
                        [
                            $el("option", { value: MODEL_SORT_DATE_CREATED }, ["Created (newest to oldest)"]),
                            $el("option", { value: "-" + MODEL_SORT_DATE_CREATED }, ["Created (oldest to newest)"]),
                            $el("option", { value: MODEL_SORT_DATE_MODIFIED }, ["Modified (newest to oldest)"]),
                            $el("option", { value: "-" + MODEL_SORT_DATE_MODIFIED }, ["Modified (oldest to newest)"]),
                            $el("option", { value: MODEL_SORT_DATE_NAME }, ["Name (A-Z)"]),
                            $el("option", { value: "-" + MODEL_SORT_DATE_NAME }, ["Name (Z-A)"]),
                        ],
                    ),
                ]),
                $el("div.row.tab-header-flex-block", [
                    $el("div.search-models", [
                        searchInput,
                        searchDropdown.element,
                    ]),
                    $el("button.icon-button", {
                        type: "button",
                        textContent: "🔍︎",
                        onclick: () => this.#modelTab_updateModelGrid(),
                    }),
                ]),
            ]),
            modelGrid,
        ];
    }

    #modelTab_updateModelGrid = () => {
        const sortValue = this.#el.modelSortSelect.value;
        const reverseSort = sortValue[0] === "-";
        const sortBy = reverseSort ? sortValue.substring(1) : sortValue;
        ModelGrid.update(
            this.#el.modelGrid,
            this.#data.models,
            this.#el.modelTypeSelect,
            this.#data.previousModelType,
            this.#el.settings,
            sortBy,
            reverseSort,
            this.#data.previousModelFilters,
            this.#el.modelContentFilter,
            this.#searchSeparator,
            this.#systemSeparator,
            this.#modelTab_showModelInfo,
        );
    }

    async #modelTab_updateModels() {
        this.#systemSeparator = await request("/model-manager/system-separator");
        this.#data.models = await request("/model-manager/models/list");
        const newModelDirectories = await request("/model-manager/models/directory-list");
        this.#data.modelDirectories.splice(0, Infinity, ...newModelDirectories); // note: do NOT create a new array
        this.#modelTab_updateModelGrid();
    }

    #modelTab_updatePreviousModelFilter = () => {
        const modelType = this.#el.modelTypeSelect.value;
        const value = this.#el.modelContentFilter.value;
        this.#data.previousModelFilters[modelType] = value;
    };

    #modelTab_updateDirectoryDropdown = () => {
        this.#modelContentFilterDirectoryDropdown.update(
            this.#data.modelDirectories,
            this.#searchSeparator,
            this.#el.modelTypeSelect.value,
        );
        this.#modelTab_updatePreviousModelFilter();
    }

    /**
     * @param {string} searchPath
     */
    #modelTab_showModelInfo = async(searchPath) => {
        const path = encodeURIComponent(searchPath);
        const info = await request(`/model-manager/model/info?path=${path}`)
        .catch((err) => {
            console.log(err);
            return null;
        });
        if (info === null) {
            return;
        }
        const infoHtml = this.#el.modelInfoContainer;
        infoHtml.innerHTML = "";
        infoHtml.dataset.path = searchPath;
        const innerHtml = [];
        const filename = info["File Name"];
        if (filename !== undefined && filename !== null && filename !== "") {
            innerHtml.push($el("h1", [filename]));
        }
        
        if (info["Preview"]) {
            const imagePath = info["Preview"]["path"];
            const imageDateModified = info["Preview"]["dateModified"];
            this.#el.modelInfoDefaultUri.dataset.noimage = imageUri(imagePath, imageDateModified);
            this.#resetModelInfoPreview();
        }
        
        innerHtml.push($el("div", [
            this.#el.modelInfoPreview,
            $el("div.row.tab-header", [
                $el("div.row.tab-header-flex-block", [
                    $el("button", {
                        textContent: "Set as Preview",
                        onclick: async(e) => {
                            const confirmation = window.confirm("Change preview image PERMANENTLY?");
                            let updatedPreview = false;
                            if (confirmation) {
                                e.target.disabled = true;
                                const container = this.#el.modelInfoContainer;
                                const path = container.dataset.path;
                                const imageUrl = getImage();
                                if (imageUrl === imageUri()) {
                                    const encodedPath = encodeURIComponent(path);
                                    updatedPreview = await request(
                                        `/model-manager/preview/delete?path=${encodedPath}`,
                                        {
                                            method: "POST",
                                            body: JSON.stringify({}),
                                        }
                                    )
                                    .then((result) => {
                                        return result["success"];
                                    })
                                    .catch((err) => {
                                        return false;
                                    });
                                }
                                else {
                                    const formData = new FormData();
                                    formData.append("path", path);
                                    const image = imageUrl[0] == "/" ? "" : imageUrl;
                                    formData.append("image", image);
                                    updatedPreview = await request(
                                        `/model-manager/preview/set`,
                                        {
                                            method: "POST",
                                            body: formData,
                                        }
                                    )
                                    .then((result) => {
                                        return result["success"];
                                    })
                                    .catch((err) => {
                                        return false;
                                    });
                                }
                                if (updatedPreview) {
                                    this.#modelTab_updateModels();
                                    this.#el.modelInfoDefaultUri.dataset.noimage = imageUri();
                                    this.#resetModelInfoPreview();
                                    this.#el.modelInfoView.style.display = "none";
                                }
                                
                                e.target.disabled = false;
                            }
                            buttonAlert(e.target, updatedPreview);
                        },
                    }),
                ]),
                $el("div.row.tab-header-flex-block", [
                    this.#el.modelInfoRadioGroup,
                ]),
            ]),
            $el("div", 
                (() => {
                    const elements = [];
                    for (const [key, value] of Object.entries(info)) {
                        if (value === undefined || value === null) {
                            continue;
                        }
                        
                        if (Array.isArray(value)) {
                            if (value.length > 0) {
                                elements.push($el("h2", [key + ":"]));
                                
                                let text = "<p>";
                                for (let i = 0; i < value.length; i++) {
                                    const v = value[i];
                                    const tag = v[0];
                                    const count = v[1];
                                    text += tag + "<span class=\"no-select\"> (" + count + ")</span>";
                                    if (i !== value.length - 1) {
                                        text += ", ";
                                    }
                                }
                                text += "</p>";
                                const div = $el("div");
                                div.innerHTML = text;
                                elements.push(div);
                            }
                        }
                        else {
                            if (key === "Notes") {
                                elements.push($el("h2", [key + ":"]));
                                const noteArea = $el("textarea.comfy-multiline-input", {
                                    value: value, 
                                    rows: 5,
                                });
                                elements.push(noteArea);
                                elements.push($el("button", {
                                    textContent: "Save Notes",
                                    onclick: (e) => {
                                        const saved = request(
                                            "/model-manager/notes/save",
                                            {
                                                method: "POST",
                                                body: JSON.stringify({
                                                    "path": this.#el.modelInfoContainer.dataset.path,
                                                    "notes": noteArea.value,
                                                }),
                                            }
                                        ).then((result) => {
                                            return result["success"];
                                        })
                                        .catch((err) => {
                                            return false;
                                        });
                                        buttonAlert(e.target, saved);
                                    },
                                }));
                            }
                            else if (key === "Preview") {
                                //
                            }
                            else {
                                if (value !== "") {
                                    elements.push($el("p", [key + ": " + value]));
                                }
                            }
                        }
                    }
                    return elements;
                })(),
            ),
        ]));
        infoHtml.append.apply(infoHtml, innerHtml);
        
        this.#el.modelInfoView.removeAttribute("style"); // remove "display: none"
        // TODO: set default value of dropdown and value to model type?
    }

    /**
     * @param {HTMLInputElement[]} settings 
     * @param {boolean} reloadData 
     */
    #setSettings(settings, reloadData) {
        const el = this.#el.settings;
        for (const [key, value] of Object.entries(settings)) {
            const setting = el[key];
            if (setting === undefined || setting === null) {
                continue;
            }
            const type = setting.type;
            switch (type) {
                case "checkbox": setting.checked = Boolean(value); break;
                case "range": setting.value = parseFloat(value); break;
                case "textarea": setting.value = value; break;
                case "number": setting.value = parseInt(value); break;
                default: console.warn("Unknown settings input type!");
            }
        }

        if (reloadData) {
            // Is this slow?
            this.#modelTab_updateModels();
        }
    }

    /**
     * @param {boolean} reloadData 
     */
    async #settingsTab_reload(reloadData) {
        const data = await request("/model-manager/settings/load");
        const settings = data["settings"];
        this.#setSettings(settings, reloadData);
        buttonAlert(this.#el.settings_reloadBtn, true);
    }

    async #settingsTab_save() {
        let settings = {};
        for (const [setting, el] of Object.entries(this.#el.settings)) {
            if (!el) { continue; } // hack
            const type = el.type;
            let value = null;
            switch (type) {
                case "checkbox": value = el.checked; break;
                case "range": value = el.value; break;
                case "textarea": value = el.value; break;
                case "number": value = el.value; break;
                default: console.warn("Unknown settings input type!");
            }
            settings[setting] = value;
        }

        const data = await request(
            "/model-manager/settings/save",
            {
                method: "POST",
                body: JSON.stringify({ "settings": settings }),
            }
        ).catch((err) => {
            return { "success": false };
        });
        const success = data["success"];
        if (success) {
            const settings = data["settings"];
            this.#setSettings(settings, true);
        }
        buttonAlert(this.#el.settings_saveBtn, success);
    }

    /**
     * @returns {HTMLElement}
     */
    #settingsTab_new() {
        const settingsTab = $el("div.model-manager-settings", [
            $el("h1", ["Settings"]),
            $el("div", [
                $el("button", {
                    $: (el) => (this.#el.settings_reloadBtn = el),
                    type: "button",
                    textContent: "Reload", // ⟳
                    onclick: () => this.#settingsTab_reload(true),
                }),
                $el("button", {
                    $: (el) => (this.#el.settings_saveBtn = el),
                    type: "button",
                    textContent: "Save", // 💾︎
                    onclick: () => this.#settingsTab_save(),
                }),
            ]),
            /*
            $el("h2", ["Window"]),
            $el("div", [
                $el("p", ["Default sidebar width"]),
                $el("input", {
                    $: (el) => (this.#el.settings["sidebar-default-width"] = el),
                    type: "number",
                    value: 0.5,
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                }),
            ]),
            $el("div", [
                $el("p", ["Default sidebar height"]),
                $el("input", {
                    $: (el) => (this.#el.settings["sidebar-default-height"] = el),
                    type: "number",
                    textContent: "Default sidebar height",
                    value: 0.5,
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                }),
            ]),
            */
            $el("h2", ["Model Search"]),
            $el("div", [
                $el("div.search-settings-text", [
                    $el("p", ["Always include in model search:"]),
                    $el("textarea.comfy-multiline-input", {
                        $: (el) => (this.#el.settings["model-search-always-append"] = el),
                        placeholder: "example: -nsfw",
                    }),
                ]),
            ]),
            $checkbox({
                $: (el) => (this.#el.settings["model-persistent-search"] = el),
                textContent: "Persistent search text across model types",
            }),
            $checkbox({
                $: (el) => (this.#el.settings["model-show-label-extensions"] = el),
                textContent: "Show model file extension in labels",
            }),
            $checkbox({
                $: (el) => (this.#el.settings["model-show-add-button"] = el),
                textContent: "Show add button",
            }),
            $checkbox({
                $: (el) => (this.#el.settings["model-show-copy-button"] = el),
                textContent: "Show copy button",
            }),
            $el("h2", ["Model Add"]),
            $checkbox({
                $: (el) => (this.#el.settings["model-add-embedding-extension"] = el),
                textContent: "Add extension to embedding",
            }),
            $checkbox({
                $: (el) => (this.#el.settings["model-add-drag-strict-on-field"] = el),
                textContent: "Strict dragging model onto a node's model field to add",
            }),
            $el("div", [
                $el("input", {
                    $: (el) => (this.#el.settings["model-add-offset"] = el),
                    type: "number",
                    step: 5,
                }),
                $el("p", ["Add model offset"]),
            ]),
        ]);
        this.#el.settingsTab = settingsTab;
        return settingsTab;
    }

    /**
     * @param {Event} e
     */
    #setSidebar(e) {
        // TODO: settings["sidebar-default-width"]
        // TODO: settings["sidebar-default-height"]
        // TODO: draggable resize?
        const button = e.target;
        const modelManager = this.element;
        const sidebarButtons = this.#el.sidebarButtons.children;

        let buttonIndex;
        for (buttonIndex = 0; buttonIndex < sidebarButtons.length; buttonIndex++) {
            if (sidebarButtons[buttonIndex] === button) {
                break;
            }
        }

        const sidebarStates = ["sidebar-left", "sidebar-top", "sidebar-bottom", "sidebar-right"];
        let stateIndex;
        for (stateIndex = 0; stateIndex < sidebarStates.length; stateIndex++) {
            const state = sidebarStates[stateIndex];
            if (modelManager.classList.contains(state)) {
                modelManager.classList.remove(state);
                break;
            }
        }

        if (stateIndex != buttonIndex) {
            const newSidebarState = sidebarStates[buttonIndex];
            modelManager.classList.add(newSidebarState);
        }
    }

    /**
     * @param {Object} info
     * @param {String[]} modelTypes
     * @param {DirectoryItem[]} modelDirectories
     * @param {String} searchSeparator
     * @param {int} id
     * @returns {HTMLDivElement}
     */
    #downloadTab_modelInfo(info, modelTypes, modelDirectories, searchSeparator, id) {
        const [el_radioGroup, el_preview, getImage, el_defaultUri, resetModelInfoPreview] = radioGroupImageSelect(
            "model-download-info-preview-model" + "-" + id,
            info["images"],
        );
        
        const el_modelTypeSelect = $el("select.model-select-dropdown", (() => {
            const options = [$el("option", { value: "" }, ["-- Model Type --"])];
            modelTypes.forEach((modelType) => {
                options.push($el("option", { value: modelType }, [modelType]));
            });
            return options;
        })());
        
        const el_saveDirectoryPath = $el("input.search-text-area", {
            type: "text",
            placeholder: this.#searchSeparator + "0",
            value: this.#searchSeparator + "0",
        });
        let searchDropdown = null;
        searchDropdown = new DirectoryDropdown(
            el_saveDirectoryPath,
            () => {
                const modelType = el_modelTypeSelect.value;
                if (modelType === "") { return; }
                searchDropdown.update(
                    modelDirectories,
                    searchSeparator,
                    modelType,
                );
            },
            () => {},
            () => {},
            searchSeparator,
            true,
        );
        
        const el_filename = $el("input.plain-text-area", {
            type: "text",
            placeholder: (() => {
                const filename = info["fileName"];
                // TODO: only remove valid model file extensions
                const i = filename.lastIndexOf(".");
                return i === - 1 ? filename : filename.substring(0, i);
            })(),
        });
        
        const filepath = info["downloadFilePath"];
        const modelInfo = $el("details.download-details", [
            $el("summary", [filepath + info["fileName"]]),
            $el("div", {
                style: { display: "flex", "flex-wrap": "wrap", gap: "16px" },
            }, [
                el_preview,
                $el("div.download-settings", [
                    $el("div", {
                        style: { "margin-top": "8px" }
                    }, [
                        $el("div.row.tab-header-flex-block", [
                            el_modelTypeSelect,
                        ]),
                        $el("div.row.tab-header-flex-block", [
                            el_saveDirectoryPath,
                            searchDropdown.element,
                        ]),
                        $el("div.row.tab-header-flex-block", [
                            $el("button.icon-button", {
                                textContent: "📥︎",
                                onclick: async (e) => {
                                    const formData = new FormData();
                                    formData.append("download", info["downloadUrl"]);
                                    formData.append("path",
                                        el_modelTypeSelect.value + 
                                        this.#searchSeparator + // NOTE: this may add multiple separators (server should handle carefully)
                                        el_saveDirectoryPath.value
                                    );
                                    formData.append("name", (() => {
                                        const filename = info["fileName"];
                                        const name = el_filename.value;
                                        if (name === "") {
                                            return filename;
                                        }
                                        const ext = MODEL_EXTENSIONS.find((ext) => {
                                            return filename.endsWith(ext);
                                        }) ?? "";
                                        return name + ext;
                                    })());
                                    formData.append("image", getImage());
                                    formData.append("overwrite", this.#el.modelInfoOverwrite.checked);
                                    e.target.disabled = true;
                                    const [success, resultText] = await request(
                                        "/model-manager/model/download",
                                        {
                                            method: "POST",
                                            body: formData,
                                        }
                                    ).then((data) => {
                                        const success = data["success"];
                                        if (!success) {
                                            console.warn(data["invalid"]);
                                        }
                                        return [success, success ? "✔" : "📥︎"];
                                    }).catch((err) => {
                                        return [false, "📥︎"];
                                    });
                                    if (success) {
                                        this.#modelTab_updateModels();
                                    }
                                    buttonAlert(e.target, success, "✔", "✖", resultText);
                                    e.target.disabled = success;
                                },
                            }),
                            el_filename,
                        ]),
                        el_radioGroup,
                    ]),
                ]),
            ]),
        ]);
        
        el_modelTypeSelect.selectedIndex = 0; // reset
        const comfyUIModelType = (
            modelTypeToComfyUiDirectory(info["details"]["fileType"]) ??
            modelTypeToComfyUiDirectory(info["modelType"]) ??
            null
        );
        if (comfyUIModelType !== undefined && comfyUIModelType !== null) {
            const modelTypeOptions = el_modelTypeSelect.children;
            for (let i = 0; i < modelTypeOptions.length; i++) {
                const option = modelTypeOptions[i];
                if (option.value === comfyUIModelType) {
                    el_modelTypeSelect.selectedIndex = i;
                    break;
                }
            }
        }
        
        return modelInfo;
    }

    async #downloadTab_search() {
        const infosHtml = this.#el.modelInfos;
        infosHtml.innerHTML = "";

        const urlText = this.#el.modelInfoUrl.value;
        const modelInfos = await (async () => {
            if (urlText.startsWith("https://civitai.com")) {
                const civitaiInfo = await civitai_getFilteredInfo(urlText);
                if (Object.keys(civitaiInfo).length === 0) {
                    return [];
                }
                const infos = [];
                const type = civitaiInfo["type"];
                civitaiInfo["versions"].forEach((version) => {
                    const images = version["images"];
                    version["files"].forEach((file) => {
                        infos.push({
                            "images": images,
                            "fileName": file["name"],
                            "modelType": type,
                            "downloadUrl": file["downloadUrl"],
                            "downloadFilePath": "",
                            "details": {
                                "fileSizeKB": file["sizeKB"],
                                "fileType": file["type"],
                                "fp": file["fp"],
                                "quant": file["size"],
                                "fileFormat": file["format"],
                            },
                        });
                    });
                });
                return infos;
            }
            if (urlText.startsWith("https://huggingface.co")) {
                const hfInfo = await huggingFace_getFilteredInfo(urlText);
                if (Object.keys(hfInfo).length === 0) {
                    return [];
                }
                const files = hfInfo["modelFiles"];
                if (files.length === 0) {
                    return [];
                }
                
                const baseDownloadUrl = hfInfo["baseDownloadUrl"];
                return hfInfo["modelFiles"].map((file) => {
                    const indexSep = file.lastIndexOf("/");
                    const filename = file.substring(indexSep + 1);
                    return {
                        "images": hfInfo["images"],
                        "fileName": filename,
                        "modelType": "",
                        "downloadUrl": baseDownloadUrl + "/" + file + "?download=true",
                        "downloadFilePath": file.substring(0, indexSep + 1),
                        "details": {
                            "fileSizeKB": undefined, // TODO: too hard?
                        },
                    };
                });
            }
            if (urlText.endsWith(".json")) {
                const indexInfo = await request(urlText).catch(() => []);
                return indexInfo.map((file) => {
                    return {
                        "images": [],
                        "fileName": file["name"],
                        "modelType": modelTypeToComfyUiDirectory(file["type"], "") ?? "",
                        "downloadUrl": file["download"],
                        "downloadFilePath": "",
                        "details": {},
                    };
                });
            }
            return [];
        })();
        
        const modelTypes = Object.keys(this.#data.models);
        const modelInfosHtml = modelInfos.filter((modelInfo) => {
            const filename = modelInfo["fileName"];
            return MODEL_EXTENSIONS.find((ext) => {
                return filename.endsWith(ext);
            }) ?? false;
        }).map((modelInfo, id) => {
            return this.#downloadTab_modelInfo(
                modelInfo,
                modelTypes,
                this.#data.modelDirectories,
                this.#searchSeparator,
                id,
            );
        });
        if (modelInfos.length === 0) {
            modelInfosHtml.push($el("div", ["No results found."]));
        }
        else {
            if (modelInfos.length === 1) {
                modelInfosHtml[0].open = true;
            }
            const label = $checkbox({
                $: (el) => { this.#el.modelInfoOverwrite = el; },
                textContent: "Overwrite Existing Files",
            });
            modelInfosHtml.unshift(label);
        }
        infosHtml.append.apply(infosHtml, modelInfosHtml);
    }
    
    /**
     * @returns {HTMLElement}
     */
    #downloadTab_new() {
        return $el("div.tab-header", [
            $el("div.row.tab-header-flex-block", [
                $el("input.search-text-area", {
                    $: (el) => (this.#el.modelInfoUrl = el),
                    type: "text",
                    placeholder: "example: https://civitai.com/models/207992/stable-video-diffusion-svd",
                    onkeydown: (e) => {
                        if (e.key === "Enter") {
                            e.stopPropagation();
                            this.#downloadTab_search();
                        }
                    },
                }),
                $el("button.icon-button", {
                    onclick: () => this.#downloadTab_search(),
                    textContent: "🔍︎",
                }),
            ]),
            $el("div.download-model-infos", {
                $: (el) => (this.#el.modelInfos = el),
            }, [
                $el("div", ["Input a URL to select a model to download."]),
            ]),
        ]);
    }
}

let instance;

/**
 * @returns {ModelManager}
 */
function getInstance() {
    if (!instance) {
        instance = new ModelManager();
    }
    return instance;
}

app.registerExtension({
    name: "Comfy.ModelManager",
    init() {
    },
    async setup() {
        $el("link", {
            parent: document.head,
            rel: "stylesheet",
            href: "./extensions/ComfyUI-Model-Manager/model-manager.css",
        });

        app.ui.menuContainer.appendChild(
            $el("button", {
                id: "comfyui-model-manager-button",
                parent: document.querySelector(".comfy-menu"),
                textContent: "Models",
                onclick: () => { getInstance().show(); },
            })
        );
    },
});
