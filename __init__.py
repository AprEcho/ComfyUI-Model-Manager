import os
import io
import pathlib
import shutil
from datetime import datetime
import sys
import copy
import importlib
import re
import base64

from aiohttp import web
import server
import urllib.parse
import urllib.request
import struct
import json
import requests
requests.packages.urllib3.disable_warnings()

import folder_paths

comfyui_model_uri = folder_paths.models_dir

extension_uri = os.path.dirname(__file__)

config_loader_path = os.path.join(extension_uri, 'config_loader.py')
config_loader_spec = importlib.util.spec_from_file_location('config_loader', config_loader_path)
config_loader = importlib.util.module_from_spec(config_loader_spec)
config_loader_spec.loader.exec_module(config_loader)

no_preview_image = os.path.join(extension_uri, "no-preview.png")
ui_settings_uri = os.path.join(extension_uri, "ui_settings.yaml")
server_settings_uri = os.path.join(extension_uri, "server_settings.yaml")

fallback_model_extensions = set([".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"]) # TODO: magic values
image_extensions = (
    ".png", # order matters
    ".webp", 
    ".jpeg", 
    ".jpg", 
    ".gif", 
    ".apng", 
)
stable_diffusion_webui_civitai_helper_image_extensions = (
    ".preview.png", # order matters
    ".preview.webp", 
    ".preview.jpeg", 
    ".preview.jpg", 
    ".preview.gif", 
    ".preview.apng", 
)
preview_extensions = ( # TODO: JavaScript does not know about this (x2 states)
    image_extensions + # order matters
    stable_diffusion_webui_civitai_helper_image_extensions
)
model_info_extension = ".txt"
#video_extensions = (".avi", ".mp4", ".webm") # TODO: Requires ffmpeg or cv2. Cache preview frame?

def split_valid_ext(s, *arg_exts):
    sl = s.lower()
    for exts in arg_exts:
        for ext in exts:
            if sl.endswith(ext.lower()):
                return (s[:-len(ext)], ext)
    return (s, "")

_folder_names_and_paths = None # dict[str, tuple[list[str], list[str]]]
def folder_paths_folder_names_and_paths(refresh = False):
    global _folder_names_and_paths
    if refresh or _folder_names_and_paths is None:
        _folder_names_and_paths = {}
        for item_name in os.listdir(comfyui_model_uri):
            item_path = os.path.join(comfyui_model_uri, item_name)
            if not os.path.isdir(item_path):
                continue
            if item_name == "configs":
                continue
            if item_name in folder_paths.folder_names_and_paths:
                dir_paths, extensions = copy.deepcopy(folder_paths.folder_names_and_paths[item_name])
            else:
                dir_paths = [item_path]
                extensions = copy.deepcopy(fallback_model_extensions)
            _folder_names_and_paths[item_name] = (dir_paths, extensions)
    return _folder_names_and_paths

def folder_paths_get_folder_paths(folder_name, refresh = False): # API function crashes querying unknown model folder
    paths = folder_paths_folder_names_and_paths(refresh)
    if folder_name in paths:
        return paths[folder_name][0]

    maybe_path = os.path.join(comfyui_model_uri, folder_name)
    if os.path.exists(maybe_path):
        return [maybe_path]
    return []

def folder_paths_get_supported_pt_extensions(folder_name, refresh = False): # Missing API function
    paths = folder_paths_folder_names_and_paths(refresh)
    if folder_name in paths:
        return paths[folder_name][1]
    model_extensions = copy.deepcopy(fallback_model_extensions)
    return model_extensions


def search_path_to_system_path(model_path):
    sep = os.path.sep
    model_path = os.path.normpath(model_path.replace("/", sep))

    isep0 = 0 if model_path[0] == sep else -1

    isep1 = model_path.find(sep, isep0 + 1)
    if isep1 == -1 or isep1 == len(model_path):
        return (None, None)

    isep2 = model_path.find(sep, isep1 + 1)
    if isep2 == -1 or isep2 - isep1 == 1:
        isep2 = len(model_path)

    model_path_type = model_path[isep0 + 1:isep1]
    paths = folder_paths_get_folder_paths(model_path_type)
    if len(paths) == 0:
        return (None, None)

    model_path_index = model_path[isep1 + 1:isep2]
    try:
        model_path_index = int(model_path_index)
    except:
        return (None, None)
    if model_path_index < 0 or model_path_index >= len(paths):
        return (None, None)

    system_path = os.path.normpath(
        paths[model_path_index] + 
        sep + 
        model_path[isep2:]
    )

    return (system_path, model_path_type)


def get_safetensor_header(path):
    try:
        with open(path, "rb") as f:
            length_of_header = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(length_of_header)
            header_json = json.loads(header_bytes)
            return header_json
    except:
        return {}


def end_swap_and_pop(x, i):
    x[i], x[-1] = x[-1], x[i]
    return x.pop(-1)


def model_type_to_dir_name(model_type):
    if model_type == "checkpoint": return "checkpoints"
    #elif model_type == "clip": return "clip"
    #elif model_type == "clip_vision": return "clip_vision"
    #elif model_type == "controlnet": return "controlnet"
    elif model_type == "diffuser": return "diffusers"
    elif model_type == "embedding": return "embeddings"
    #elif model_type== "gligen": return "gligen"
    elif model_type == "hypernetwork": return "hypernetworks"
    elif model_type == "lora": return "loras"
    #elif model_type == "style_models": return "style_models"
    #elif model_type == "unet": return "unet"
    elif model_type == "upscale_model": return "upscale_models"
    #elif model_type == "vae": return "vae"
    #elif model_type == "vae_approx": return "vae_approx"
    else: return model_type


def ui_rules():
    Rule = config_loader.Rule
    return [
        Rule("sidebar-default-height", 0.5, float, 0.0, 1.0),
        Rule("sidebar-default-width", 0.5, float, 0.0, 1.0),
        Rule("model-search-always-append", "", str),
        Rule("model-persistent-search", True, bool),
        Rule("model-show-label-extensions", False, bool),
        Rule("model-preview-fallback-search-safetensors-thumbnail", False, bool),
        Rule("model-show-add-button", True, bool),
        Rule("model-show-copy-button", True, bool),
        Rule("model-add-embedding-extension", False, bool),
        Rule("model-add-drag-strict-on-field", False, bool),
        Rule("model-add-offset", 25, int),
    ]


def server_rules():
    Rule = config_loader.Rule
    return [
        #Rule("model_extension_download_whitelist", [".safetensors"], list),
        Rule("civitai_api_key", "", str),
        Rule("huggingface_api_key", "", str),
    ]
server_settings = config_loader.yaml_load(server_settings_uri, server_rules())
config_loader.yaml_save(server_settings_uri, server_rules(), server_settings)

@server.PromptServer.instance.routes.get("/model-manager/settings/load")
async def load_ui_settings(request):
    rules = ui_rules()
    settings = config_loader.yaml_load(ui_settings_uri, rules)
    return web.json_response({ "settings": settings })


@server.PromptServer.instance.routes.post("/model-manager/settings/save")
async def save_ui_settings(request):
    body = await request.json()
    settings = body.get("settings")
    rules = ui_rules()
    validated_settings = config_loader.validated(rules, settings)
    success = config_loader.yaml_save(ui_settings_uri, rules, validated_settings)
    return web.json_response({
        "success": success,
        "settings": validated_settings if success else "",
    })


@server.PromptServer.instance.routes.get("/model-manager/preview/get")
async def get_model_preview(request):
    uri = request.query.get("uri")

    image_path = no_preview_image
    image_type = "png"
    image_data = None
    if uri != "no-preview":
        sep = os.path.sep
        uri = uri.replace("/" if sep == "\\" else "/", sep)
        path, _ = search_path_to_system_path(uri)
        head, extension = split_valid_ext(path, preview_extensions)
        if os.path.exists(path):
            image_type = extension.rsplit(".", 1)[1]
            image_path = path
        elif os.path.exists(head) and head.endswith(".safetensors"):
            image_type = extension.rsplit(".", 1)[1]
            header = get_safetensor_header(head)
            metadata = header.get("__metadata__", None)
            if metadata is not None:
                thumbnail = metadata.get("modelspec.thumbnail", None)
                if thumbnail is not None:
                    image_data = thumbnail.split(',')[1]
                    image_data = base64.b64decode(image_data)

    if image_data == None:
        with open(image_path, "rb") as file:
            image_data = file.read()

    return web.Response(body=image_data, content_type="image/" + image_type)


@server.PromptServer.instance.routes.get("/model-manager/image/extensions")
async def get_image_extensions(request):
    return web.json_response(image_extensions)


def download_model_preview(formdata):
    path = formdata.get("path", None)
    if type(path) is not str:
        raise ("Invalid path!")
    path, model_type = search_path_to_system_path(path)
    model_type_extensions = folder_paths_get_supported_pt_extensions(model_type)
    path_without_extension, _ = split_valid_ext(path, model_type_extensions)

    overwrite = formdata.get("overwrite", "true").lower()
    overwrite = True if overwrite == "true" else False

    image = formdata.get("image", None)
    if type(image) is str:
        _, image_extension = split_valid_ext(image, image_extensions) # TODO: doesn't work for https://civitai.com/images/...
        if image_extension == "":
            raise ValueError("Invalid image type!")
        image_path = path_without_extension + image_extension
        download_file(image, image_path, overwrite)
    else:
        content_type = image.content_type
        if not content_type.startswith("image/"):
            raise ("Invalid content type!")
        image_extension = "." + content_type[len("image/"):]
        if image_extension not in image_extensions:
            raise ("Invalid extension!")

        image_path = path_without_extension + image_extension
        if not overwrite and os.path.isfile(image_path):
            raise ("Image already exists!")
        file: io.IOBase = image.file
        image_data = file.read()
        with open(image_path, "wb") as f:
            f.write(image_data)

    delete_same_name_files(path_without_extension, preview_extensions, image_extension)


@server.PromptServer.instance.routes.post("/model-manager/preview/set")
async def set_model_preview(request):
    formdata = await request.post()
    try:
        download_model_preview(formdata)
        return web.json_response({ "success": True })
    except ValueError as e:
        print(e, file=sys.stderr, flush=True)
        return web.json_response({ "success": False })


@server.PromptServer.instance.routes.post("/model-manager/preview/delete")
async def delete_model_preview(request):
    model_path = request.query.get("path", None)
    if model_path is None:
        return web.json_response({ "success": False })
    model_path = urllib.parse.unquote(model_path)

    model_path, model_type = search_path_to_system_path(model_path)
    model_extensions = folder_paths_get_supported_pt_extensions(model_type)
    path_and_name, _ = split_valid_ext(model_path, model_extensions)
    delete_same_name_files(path_and_name, preview_extensions)

    return web.json_response({ "success": True })


@server.PromptServer.instance.routes.get("/model-manager/models/list")
async def get_model_list(request):
    use_safetensor_thumbnail = (
        config_loader.yaml_load(ui_settings_uri, ui_rules())
        .get("model-preview-fallback-search-safetensors-thumbnail", False)
    )

    model_types = os.listdir(comfyui_model_uri)
    model_types.remove("configs")
    model_types.sort()

    models = {}
    for model_type in model_types:
        model_extensions = tuple(folder_paths_get_supported_pt_extensions(model_type))
        file_infos = []
        for base_path_index, model_base_path in enumerate(folder_paths_get_folder_paths(model_type)):
            if not os.path.exists(model_base_path): # TODO: Bug in main code? ("ComfyUI\output\checkpoints", "ComfyUI\output\clip", "ComfyUI\models\t2i_adapter", "ComfyUI\output\vae")
                continue
            for cwd, subdirs, files in os.walk(model_base_path):
                dir_models = []
                dir_images = []

                for file in files:
                    if file.lower().endswith(model_extensions):
                        dir_models.append(file)
                    elif file.lower().endswith(preview_extensions):
                        dir_images.append(file)

                for model in dir_models:
                    model_name, model_ext = split_valid_ext(model, model_extensions)
                    image = None
                    image_modified = None
                    for ext in preview_extensions: # order matters
                        for iImage in range(len(dir_images)-1, -1, -1):
                            image_name = dir_images[iImage]
                            if not image_name.lower().endswith(ext.lower()):
                                continue
                            image_name = image_name[:-len(ext)]
                            if model_name == image_name:
                                image = end_swap_and_pop(dir_images, iImage)
                                img_abs_path = os.path.join(cwd, image)
                                image_modified = pathlib.Path(img_abs_path).stat().st_mtime_ns
                                break
                        if image is not None:
                            break
                    abs_path = os.path.join(cwd, model)
                    stats = pathlib.Path(abs_path).stat()
                    model_modified = stats.st_mtime_ns
                    model_created = stats.st_ctime_ns
                    if use_safetensor_thumbnail and image is None and model_ext == ".safetensors":
                        # try to fallback on safetensor embedded thumbnail
                        header = get_safetensor_header(abs_path)
                        metadata = header.get("__metadata__", None)
                        if metadata is not None:
                            thumbnail = metadata.get("modelspec.thumbnail", None)
                            if thumbnail is not None:
                                i0 = thumbnail.find("/") + 1
                                i1 = thumbnail.find(";")
                                image_ext = "." + thumbnail[i0:i1]
                                if image_ext in image_extensions:
                                    image = model + image_ext
                                    image_modified = model_modified
                    rel_path = "" if cwd == model_base_path else os.path.relpath(cwd, model_base_path)
                    info = (model, image, base_path_index, rel_path, model_modified, model_created, image_modified)
                    file_infos.append(info)
        file_infos.sort(key=lambda tup: tup[4], reverse=True) # TODO: remove sort; sorted on client

        model_items = []
        for model, image, base_path_index, rel_path, model_modified, model_created, image_modified in file_infos:
            item = {
                "name": model,
                "path": "/" + os.path.join(model_type, str(base_path_index), rel_path, model).replace(os.path.sep, "/"), # relative logical path
                #"systemPath": os.path.join(rel_path, model), # relative system path (less information than "search path")
                "dateModified": model_modified,
                "dateCreated": model_created,
                #"dateLastUsed": "", # TODO: track server-side, send increment client-side
                #"countUsed": 0, # TODO: track server-side, send increment client-side
            }
            if image is not None:
                raw_post = os.path.join(model_type, str(base_path_index), rel_path, image)
                item["preview"] = {
                    "path": urllib.parse.quote_plus(raw_post),
                    "dateModified": urllib.parse.quote_plus(str(image_modified)),
                }
            model_items.append(item)

        models[model_type] = model_items

    return web.json_response(models)


def linear_directory_hierarchy(refresh = False):
    model_paths = folder_paths_folder_names_and_paths(refresh)
    dir_list = []
    dir_list.append({ "name": "", "childIndex": 1, "childCount": len(model_paths) })
    for model_dir_name, (model_dirs, _) in model_paths.items():
        dir_list.append({ "name": model_dir_name, "childIndex": None, "childCount": len(model_dirs) })
    for model_dir_index, (_, (model_dirs, extension_whitelist)) in enumerate(model_paths.items()):
        model_dir_child_index = len(dir_list)
        dir_list[model_dir_index + 1]["childIndex"] = model_dir_child_index
        for dir_path_index, dir_path in enumerate(model_dirs):
            dir_list.append({ "name": str(dir_path_index), "childIndex": None, "childCount": None })
        for dir_path_index, dir_path in enumerate(model_dirs):
            if not os.path.exists(dir_path) or os.path.isfile(dir_path):
                continue
            
            #dir_list.append({ "name": str(dir_path_index), "childIndex": None, "childCount": 0 })
            dir_stack = [(dir_path, model_dir_child_index + dir_path_index)]
            while len(dir_stack) > 0: # DEPTH-FIRST
                dir_path, dir_index = dir_stack.pop()
                
                dir_items = os.listdir(dir_path)
                dir_items = sorted(dir_items, key=str.casefold)
                
                dir_child_count = 0
                
                # TODO: sort content of directory: alphabetically
                # TODO: sort content of directory: files first
                
                subdirs = []
                for item_name in dir_items: # BREADTH-FIRST
                    item_path = os.path.join(dir_path, item_name)
                    if os.path.isdir(item_path):
                        # dir
                        subdir_index = len(dir_list) # this must be done BEFORE `dir_list.append`
                        subdirs.append((item_path, subdir_index))
                        dir_list.append({ "name": item_name, "childIndex": None, "childCount": 0 })
                        dir_child_count += 1
                    else:
                        # file
                        if extension_whitelist is None or split_valid_ext(item_name, extension_whitelist)[1] != "":
                            dir_list.append({ "name": item_name })
                            dir_child_count += 1
                if dir_child_count > 0:
                    dir_list[dir_index]["childIndex"] = len(dir_list) - dir_child_count
                dir_list[dir_index]["childCount"] = dir_child_count
                subdirs.reverse()
                for dir_path, subdir_index in subdirs:
                    dir_stack.append((dir_path, subdir_index))
    return dir_list


@server.PromptServer.instance.routes.get("/model-manager/models/directory-list")
async def get_directory_list(request):
    #body = await request.json()
    dir_list = linear_directory_hierarchy(True)
    #json.dump(dir_list, sys.stdout, indent=4)
    return web.json_response(dir_list)


def download_file(url, filename, overwrite):
    if not overwrite and os.path.isfile(filename):
        raise ValueError("File already exists!")

    filename_temp = filename + ".download"

    def_headers = {
        "User-Agent": "Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    }

    if url.startswith("https://civitai.com/"):
        api_key = server_settings["civitai_api_key"]
        if (api_key != ""):
            def_headers["Authorization"] = f"Bearer {api_key}"
            url += "&" if "?" in url else "?" # not the most robust solution
            url += f"token={api_key}" # TODO: Authorization didn't work in the header
    elif url.startswith("https://huggingface.co/"):
        api_key = server_settings["huggingface_api_key"]
        if api_key != "":
            def_headers["Authorization"] = f"Bearer {api_key}"
    rh = requests.get(url=url, stream=True, verify=False, headers=def_headers, proxies=None, allow_redirects=False)
    if not rh.ok:
        raise ValueError(
            "Unable to download! Request header status code: " + 
            str(rh.status_code)
        )

    downloaded_size = 0
    if rh.status_code == 200 and os.path.exists(filename_temp):
        downloaded_size = os.path.getsize(filename_temp)

    headers = {"Range": "bytes=%d-" % downloaded_size}
    headers["User-Agent"] = def_headers["User-Agent"]
    
    r = requests.get(url=url, stream=True, verify=False, headers=headers, proxies=None, allow_redirects=False)
    if rh.status_code == 307 and r.status_code == 307:
        # Civitai redirect
        redirect_url = r.content.decode("utf-8")
        if not redirect_url.startswith("http"):
            # Civitai requires login (NSFW or user-required)
            # TODO: inform user WHY download failed
            raise ValueError("Unable to download from Civitai! Redirect url: " + str(redirect_url))
        download_file(redirect_url, filename, overwrite)
        return
    if rh.status_code == 302 and r.status_code == 302:
        # HuggingFace redirect
        redirect_url = r.content.decode("utf-8")
        redirect_url_index = redirect_url.find("http")
        if redirect_url_index == -1:
            raise ValueError("Unable to download from HuggingFace! Redirect url: " + str(redirect_url))
        download_file(redirect_url[redirect_url_index:], filename, overwrite)
        return
    elif rh.status_code == 200 and r.status_code == 206:
        # Civitai download link
        pass

    total_size = int(rh.headers.get("Content-Length", 0)) # TODO: pass in total size earlier

    print("Download file: " + filename)
    if total_size != 0:
        print("Download file size: " + str(total_size))

    mode = "wb" if overwrite else "ab"
    with open(filename_temp, mode) as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk is not None:
                downloaded_size += len(chunk)
                f.write(chunk)
                f.flush()

                if total_size != 0:
                    fraction = 1 if downloaded_size == total_size else downloaded_size / total_size
                    progress = int(50 * fraction)
                    sys.stdout.reconfigure(encoding="utf-8")
                    sys.stdout.write(
                        "\r[%s%s] %d%%"
                        % (
                            "-" * progress,
                            " " * (50 - progress),
                            100 * fraction,
                        )
                    )
                    sys.stdout.flush()
    print()

    if overwrite and os.path.isfile(filename):
        os.remove(filename)
    os.rename(filename_temp, filename)


@server.PromptServer.instance.routes.get("/model-manager/model/info")
async def get_model_info(request):
    model_path = request.query.get("path", None)
    if model_path is None:
        return web.json_response({ "success": False })
    model_path = urllib.parse.unquote(model_path)

    abs_path, model_type = search_path_to_system_path(model_path)
    if abs_path is None:
        return web.json_response({})

    info = {}
    comfyui_directory, name = os.path.split(model_path)
    info["File Name"] = name
    info["File Directory"] = comfyui_directory
    info["File Size"] = str(os.path.getsize(abs_path)) + " bytes"
    stats = pathlib.Path(abs_path).stat()
    date_format = "%Y-%m-%d %H:%M:%S"
    date_modified = datetime.fromtimestamp(stats.st_mtime).strftime(date_format)
    info["Date Modified"] = date_modified
    info["Date Created"] = datetime.fromtimestamp(stats.st_ctime).strftime(date_format)

    model_extensions = folder_paths_get_supported_pt_extensions(model_type)
    abs_name , _ = split_valid_ext(abs_path, model_extensions)

    for extension in preview_extensions:
        maybe_preview = abs_name + extension
        if os.path.isfile(maybe_preview):
            preview_path, _ = split_valid_ext(model_path, model_extensions)
            preview_modified = pathlib.Path(maybe_preview).stat().st_mtime_ns
            info["Preview"] = {
                "path": urllib.parse.quote_plus(preview_path + extension),
                "dateModified": urllib.parse.quote_plus(str(preview_modified)),
            }
            break

    header = get_safetensor_header(abs_path)
    metadata = header.get("__metadata__", None)
    #json.dump(metadata, sys.stdout, indent=4)
    #print()

    if metadata is not None and info.get("Preview", None) is None:
        thumbnail = metadata.get("modelspec.thumbnail")
        if thumbnail is not None:
            i0 = thumbnail.find("/") + 1
            i1 = thumbnail.find(";", i0)
            thumbnail_extension = "." + thumbnail[i0:i1]
            if thumbnail_extension in image_extensions:
                info["Preview"] = {
                    "path": request.query["path"] + thumbnail_extension,
                    "dateModified": date_modified,
                }

    if metadata is not None:
        train_end = metadata.get("modelspec.date", "").replace("T", " ")
        train_start = metadata.get("ss_training_started_at", "")
        if train_start != "":
            try:
                train_start = float(train_start)
                train_start = datetime.fromtimestamp(train_start).strftime(date_format)
            except:
                train_start = ""
        info["Date Trained"] = (
            train_start + 
            (" ... " if train_start != "" and train_end != "" else "") + 
            train_end
        )

        info["Base Training Model"] = metadata.get("ss_sd_model_name", "")
        info["Base Model"] = metadata.get("ss_base_model_version", "")
        info["Architecture"] = metadata.get("modelspec.architecture", "")
        info["Network Dimension"] = metadata.get("ss_network_dim", "") # features trained
        info["Network Alpha"] = metadata.get("ss_network_alpha", "") # trained features applied
        info["Model Sampling Type"] = metadata.get("modelspec.prediction_type", "")
        clip_skip = metadata.get("ss_clip_skip", "")
        if clip_skip == "None" or clip_skip == "1": # assume 1 means no clip skip
            clip_skip = ""
        info["Clip Skip"] = clip_skip

        # it is unclear what these are
        #info["Hash SHA256"] = metadata.get("modelspec.hash_sha256", "")
        #info["SSHS Model Hash"] = metadata.get("sshs_model_hash", "")
        #info["SSHS Legacy Hash"] = metadata.get("sshs_legacy_hash", "")
        #info["New SD Model Hash"] = metadata.get("ss_new_sd_model_hash", "")

        #info["Output Name"] = metadata.get("ss_output_name", "")
        #info["Title"] = metadata.get("modelspec.title", "")
        info["Author"] = metadata.get("modelspec.author", "")
        info["License"] = metadata.get("modelspec.license", "")

    if metadata is not None:
        training_comment = metadata.get("ss_training_comment", "")
        info["Description"] = (
            metadata.get("modelspec.description", "") + 
            "\n\n" + 
            metadata.get("modelspec.usage_hint", "") + 
            "\n\n" + 
            training_comment if training_comment != "None" else ""
        ).strip()

    info_text_file = abs_name + model_info_extension
    notes = ""
    if os.path.isfile(info_text_file):
        with open(info_text_file, 'r', encoding="utf-8") as f:
            notes = f.read()
    info["Notes"] = notes

    if metadata is not None:
        img_buckets = metadata.get("ss_bucket_info", "{}")
        if type(img_buckets) is str:
            img_buckets = json.loads(img_buckets)
        resolutions = {}
        if img_buckets is not None:
            buckets = img_buckets.get("buckets", {})
            for resolution in buckets.values():
                dim = resolution["resolution"]
                x, y = dim[0], dim[1]
                count = resolution["count"]
                resolutions[str(x) + "x" + str(y)] = count
        resolutions = list(resolutions.items())
        resolutions.sort(key=lambda x: x[1], reverse=True)
        info["Bucket Resolutions"] = resolutions

        dir_tags = metadata.get("ss_tag_frequency", "{}")
        if type(dir_tags) is str:
            dir_tags = json.loads(dir_tags)
        tags = {}
        for train_tags in dir_tags.values():
            for tag, count in train_tags.items():
                tags[tag] = tags.get(tag, 0) + count
        tags = list(tags.items())
        tags.sort(key=lambda x: x[1], reverse=True)
        info["Tags"] = tags

    return web.json_response(info)


@server.PromptServer.instance.routes.get("/model-manager/system-separator")
async def get_system_separator(request):
    return web.json_response(os.path.sep)


@server.PromptServer.instance.routes.post("/model-manager/model/download")
async def download_model(request):
    formdata = await request.post()
    result = {
        "success": False,
        "invalid": None,
    }

    overwrite = formdata.get("overwrite", "false").lower()
    overwrite = True if overwrite == "true" else False

    model_path = formdata.get("path", "/0")
    directory, model_type = search_path_to_system_path(model_path)
    if directory is None:
        result["invalid"] = "path"
        return web.json_response(result)

    download_uri = formdata.get("download")
    if download_uri is None:
        result["invalid"] = "download"
        return web.json_response(result)

    name = formdata.get("name")
    model_extensions = folder_paths_get_supported_pt_extensions(model_type)
    _, model_extension = split_valid_ext(name, model_extensions)
    if model_extension == "":
        result["invalid"] = "name"
        return web.json_response(result)
    file_name = os.path.join(directory, name)
    try:
        download_file(download_uri, file_name, overwrite)
    except Exception as e:
        print(e, file=sys.stderr, flush=True)
        result["invalid"] = "model"
        return web.json_response(result)

    image = formdata.get("image")
    if image is not None and image != "":
        try:
            download_model_preview({
                "path": model_path + os.sep + name,
                "image": image,
                "overwrite": formdata.get("overwrite"),
            })
        except Exception as e:
            print(e, file=sys.stderr, flush=True)
            result["invalid"] = "preview"

    result["success"] = True
    return web.json_response(result)


@server.PromptServer.instance.routes.post("/model-manager/model/move")
async def move_model(request):
    body = await request.json()

    old_file = body.get("oldFile", None)
    if old_file is None:
        return web.json_response({ "success": False })
    old_file, old_model_type = search_path_to_system_path(old_file)
    if not os.path.isfile(old_file):
        return web.json_response({ "success": False })
    old_model_extensions = folder_paths_get_supported_pt_extensions(old_model_type)
    old_file_without_extension, model_extension = split_valid_ext(old_file, old_model_extensions)
    if model_extension == "":
        # cannot move arbitrary files
        return web.json_response({ "success": False })

    new_file = body.get("newFile", None)
    if new_file is None or new_file == "":
        # cannot have empty name
        return web.json_response({ "success": False })
    new_file, new_model_type = search_path_to_system_path(new_file)
    if not new_file.endswith(model_extension):
        return web.json_response({ "success": False })
    if os.path.isfile(new_file):
        # cannot overwrite existing file
        return web.json_response({ "success": False })
    new_model_extensions = folder_paths_get_supported_pt_extensions(new_model_type)
    new_file_without_extension, new_model_extension = split_valid_ext(new_file, new_model_extensions)
    if model_extension != new_model_extension:
        # cannot change extension
        return web.json_response({ "success": False })
    new_file_dir, _ = os.path.split(new_file)
    if not os.path.isdir(new_file_dir):
        return web.json_response({ "success": False })

    if old_file == new_file:
        return web.json_response({ "success": False })
    try:
        shutil.move(old_file, new_file)
    except ValueError as e:
        print(e, file=sys.stderr, flush=True)
        return web.json_response({ "success": False })

    # TODO: this could overwrite existing files in destination...
    for extension in preview_extensions + (model_info_extension,):
        old_file = old_file_without_extension + extension
        if os.path.isfile(old_file):
            try:
                shutil.move(old_file, new_file_without_extension + extension)
            except ValueError as e:
                print(e, file=sys.stderr, flush=True)

    return web.json_response({ "success": True })


def delete_same_name_files(path_without_extension, extensions, keep_extension=None):
    for extension in extensions:
        if extension == keep_extension: continue
        file = path_without_extension + extension
        if os.path.isfile(file):
            os.remove(file)


@server.PromptServer.instance.routes.post("/model-manager/model/delete")
async def delete_model(request):
    result = { "success": False }

    model_path = request.query.get("path", None)
    if model_path is None:
        return web.json_response(result)
    model_path = urllib.parse.unquote(model_path)
    model_path, model_type = search_path_to_system_path(model_path)
    if model_path is None:
        return web.json_response(result)

    model_extensions = folder_paths_get_supported_pt_extensions(model_type)
    path_and_name, model_extension = split_valid_ext(model_path, model_extensions)
    if model_extension == "":
        # cannot delete arbitrary files
        return web.json_response(result)

    if os.path.isfile(model_path):
        os.remove(model_path)
        result["success"] = True

        delete_same_name_files(path_and_name, preview_extensions)
        delete_same_name_files(path_and_name, (model_info_extension,))

    return web.json_response(result)


@server.PromptServer.instance.routes.post("/model-manager/notes/save")
async def set_notes(request):
    body = await request.json()

    text = body.get("notes", None)
    if type(text) is not str:
        return web.json_response({ "success": False })

    model_path = body.get("path", None)
    if type(model_path) is not str:
        return web.json_response({ "success": False })
    model_path, model_type = search_path_to_system_path(model_path)
    model_extensions = folder_paths_get_supported_pt_extensions(model_type)
    file_path_without_extension, _ = split_valid_ext(model_path, model_extensions)
    filename = os.path.normpath(file_path_without_extension + model_info_extension)
    if text.isspace() or text == "":
        if os.path.exists(filename):
            os.remove(filename)
    else:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        except ValueError as e:
            print(e, file=sys.stderr, flush=True)
            web.json_response({ "success": False })

    return web.json_response({ "success": True })


WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS"]
