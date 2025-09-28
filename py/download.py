import os
import uuid
import time
import requests
import base64
import asyncio


import folder_paths


from typing import Callable, Awaitable, Any, Literal, Union, Optional
from dataclasses import dataclass
from aiohttp import web


from . import config
from . import utils
from . import thread


@dataclass
class TaskStatus:
    taskId: str
    type: str
    fullname: str
    preview: str
    status: Literal["pause", "waiting", "doing"] = "pause"
    platform: Union[str, None] = None
    downloadedSize: float = 0
    totalSize: float = 0
    progress: float = 0
    bps: float = 0
    error: Optional[str] = None

    def __init__(self, **kwargs):
        self.taskId = kwargs.get("taskId", None)
        self.type = kwargs.get("type", None)
        self.fullname = kwargs.get("fullname", None)
        self.preview = kwargs.get("preview", None)
        self.status = kwargs.get("status", "pause")
        self.platform = kwargs.get("platform", None)
        self.downloadedSize = kwargs.get("downloadedSize", 0)
        self.totalSize = kwargs.get("totalSize", 0)
        self.progress = kwargs.get("progress", 0)
        self.bps = kwargs.get("bps", 0)
        self.error = kwargs.get("error", None)

    def to_dict(self):
        return {
            "taskId": self.taskId,
            "type": self.type,
            "fullname": self.fullname,
            "preview": self.preview,
            "status": self.status,
            "platform": self.platform,
            "downloadedSize": self.downloadedSize,
            "totalSize": self.totalSize,
            "progress": self.progress,
            "bps": self.bps,
            "error": self.error,
        }


@dataclass
class TaskContent:
    type: str
    pathIndex: int
    fullname: str
    description: str
    downloadPlatform: str
    downloadUrl: str
    sizeBytes: float
    hashes: Optional[dict[str, str]] = None

    def __init__(self, **kwargs):
        self.type = kwargs.get("type", None)
        self.pathIndex = int(kwargs.get("pathIndex", 0))
        self.fullname = kwargs.get("fullname", None)
        self.description = kwargs.get("description", None)
        self.downloadPlatform = kwargs.get("downloadPlatform", None)
        self.downloadUrl = kwargs.get("downloadUrl", None)
        self.sizeBytes = float(kwargs.get("sizeBytes", 0))
        self.hashes = kwargs.get("hashes", None)

    def to_dict(self):
        return {
            "type": self.type,
            "pathIndex": self.pathIndex,
            "fullname": self.fullname,
            "description": self.description,
            "downloadPlatform": self.downloadPlatform,
            "downloadUrl": self.downloadUrl,
            "sizeBytes": self.sizeBytes,
            "hashes": self.hashes,
        }


class ApiKey:

    __store: dict[str, str] = {}

    def __init__(self):
        self.__cache_file = os.path.join(config.extension_uri, "private.key")

    def init(self, request):
        # Try to migrate api key from user setting
        if not os.path.exists(self.__cache_file):
            self.__store = {
                "civitai": utils.get_setting_value(request, "api_key.civitai"),
                "huggingface": utils.get_setting_value(request, "api_key.huggingface"),
            }
            self.__update__()
            # Remove api key from user setting
            utils.set_setting_value(request, "api_key.civitai", None)
            utils.set_setting_value(request, "api_key.huggingface", None)
        self.__store = utils.load_dict_pickle_file(self.__cache_file)
        # Desensitization returns
        result: dict[str, str] = {}
        for key in self.__store:
            v = self.__store[key]
            if v is not None:
                result[key] = v[:4] + "****" + v[-4:]
        return result

    def get_value(self, key: str):
        return self.__store.get(key, None)

    def set_value(self, key: str, value: str):
        self.__store[key] = value
        self.__update__()

    def __update__(self):
        utils.save_dict_pickle_file(self.__cache_file, self.__store)


class ModelDownload:
    def __init__(self):
        self.api_key = ApiKey()

    def add_routes(self, routes):
        @routes.post("/model-manager/download/init")
        async def init_download(request):
            """
            Init download setting.
            """
            result = self.api_key.init(request)
            return web.json_response({"success": True, "data": result})

        @routes.post("/model-manager/download/setting")
        async def set_download_setting(request):
            """
            Set download setting.
            """
            json_data = await request.json()
            key = json_data.get("key", None)
            value = json_data.get("value", None)
            value = base64.b64decode(value).decode("utf-8") if value is not None else None
            self.api_key.set_value(key, value)
            return web.json_response({"success": True})

        @routes.get("/model-manager/download/task")
        async def scan_download_tasks(request):
            """
            Read download task list.
            """
            try:
                result = await self.scan_model_download_task_list()
                return web.json_response({"success": True, "data": result})
            except Exception as e:
                error_msg = f"Read download task list failed: {e}"
                utils.print_error(error_msg)
                return web.json_response({"success": False, "error": error_msg})

        @routes.put("/model-manager/download/{task_id}")
        async def resume_download_task(request):
            """
            Toggle download task status.
            """
            try:
                task_id = request.match_info.get("task_id", None)
                if task_id is None:
                    raise web.HTTPBadRequest(reason="Invalid task id")
                json_data = await request.json()
                status = json_data.get("status", None)
                if status == "pause":
                    await self.pause_model_download_task(task_id)
                elif status == "resume":
                    await self.download_model(task_id, request)
                else:
                    raise web.HTTPBadRequest(reason="Invalid status")

                return web.json_response({"success": True})
            except Exception as e:
                error_msg = f"Resume download task failed: {str(e)}"
                utils.print_error(error_msg)
                return web.json_response({"success": False, "error": error_msg})

        @routes.delete("/model-manager/download/{task_id}")
        async def delete_model_download_task(request):
            """
            Delete download task.
            """
            task_id = request.match_info.get("task_id", None)
            try:
                await self.delete_model_download_task(task_id)
                return web.json_response({"success": True})
            except Exception as e:
                error_msg = f"Delete download task failed: {str(e)}"
                utils.print_error(error_msg)
                return web.json_response({"success": False, "error": error_msg})

        @routes.post("/model-manager/model")
        async def create_model(request):
            """
            Create a new model.

            request body: x-www-form-urlencoded
            - type: model type.
            - pathIndex: index of the model folders.
            - fullname: filename that relative to the model folder.
            - previewFile: preview file.
            - description: description.
            - downloadPlatform: download platform.
            - downloadUrl: download url.
            - hash: a JSON string containing the hash value of the downloaded model.
            """
            task_data = await request.post()
            task_data = dict(task_data)
            try:
                task_id = await self.create_model_download_task(task_data, request)
                return web.json_response({"success": True, "data": {"taskId": task_id}})
            except Exception as e:
                error_msg = f"Create model download task failed: {str(e)}"
                utils.print_error(error_msg)
                return web.json_response({"success": False, "error": error_msg})

    download_model_task_status: dict[str, TaskStatus] = {}

    download_thread_pool = thread.DownloadThreadPool()

    def set_task_content(self, task_id: str, task_content: Union[TaskContent, dict]):
        download_path = utils.get_download_path()
        task_file_path = utils.join_path(download_path, f"{task_id}.task")
        utils.save_dict_pickle_file(task_file_path, task_content)

    def get_task_content(self, task_id: str):
        download_path = utils.get_download_path()
        task_file = utils.join_path(download_path, f"{task_id}.task")
        if not os.path.isfile(task_file):
            raise RuntimeError(f"Task {task_id} not found")
        task_content = utils.load_dict_pickle_file(task_file)
        if isinstance(task_content, TaskContent):
            return task_content
        return TaskContent(**task_content)

    def get_task_status(self, task_id: str):
        task_status = self.download_model_task_status.get(task_id, None)

        if task_status is None:
            download_path = utils.get_download_path()
            task_content = self.get_task_content(task_id)
            download_file = utils.join_path(download_path, f"{task_id}.download")
            download_size = 0
            if os.path.exists(download_file):
                download_size = os.path.getsize(download_file)

            total_size = task_content.sizeBytes
            task_status = TaskStatus(
                taskId=task_id,
                type=task_content.type,
                fullname=task_content.fullname,
                preview=utils.get_model_preview_name(download_file),
                platform=task_content.downloadPlatform,
                downloadedSize=download_size,
                totalSize=task_content.sizeBytes,
                progress=download_size / total_size * 100 if total_size > 0 else 0,
            )

            self.download_model_task_status[task_id] = task_status

        return task_status

    def delete_task_status(self, task_id: str):
        self.download_model_task_status.pop(task_id, None)

    async def scan_model_download_task_list(self):
        """
        Scan the download directory and send the task list to the client.
        """
        download_dir = utils.get_download_path()
        task_files = utils.search_files(download_dir)
        task_files = folder_paths.filter_files_extensions(task_files, [".task"])
        task_files = sorted(
            task_files,
            key=lambda x: os.stat(utils.join_path(download_dir, x)).st_ctime,
            reverse=True,
        )
        task_list: list[dict] = []
        for task_file in task_files:
            task_id = task_file.replace(".task", "")
            task_status = self.get_task_status(task_id)
            task_list.append(task_status.to_dict())

        return task_list

    async def create_model_download_task(self, task_data: dict, request):
        """
        Creates a download task for the given data.
        """
        model_type = task_data.get("type", None)
        path_index = int(task_data.get("pathIndex", None))
        fullname = task_data.get("fullname", None)

        model_path = utils.get_full_path(model_type, path_index, fullname)
        # Check if the model path is valid
        if os.path.exists(model_path):
            raise RuntimeError(f"File already exists: {model_path}")

        download_path = utils.get_download_path()

        task_id = uuid.uuid4().hex
        task_path = utils.join_path(download_path, f"{task_id}.task")
        if os.path.exists(task_path):
            raise RuntimeError(f"Task {task_id} already exists")
        download_platform = task_data.get("downloadPlatform", None)

        try:
            preview_file = task_data.pop("previewFile", None)
            utils.save_model_preview(task_path, preview_file, download_platform)
            self.set_task_content(task_id, task_data)
            task_status = TaskStatus(
                taskId=task_id,
                type=model_type,
                fullname=fullname,
                preview=utils.get_model_preview_name(task_path),
                platform=download_platform,
                totalSize=float(task_data.get("sizeBytes", 0)),
            )
            self.download_model_task_status[task_id] = task_status
            await utils.send_json("create_download_task", task_status.to_dict())
        except Exception as e:
            await self.delete_model_download_task(task_id)
            raise RuntimeError(str(e)) from e

        await self.download_model(task_id, request)
        return task_id

    async def pause_model_download_task(self, task_id: str):
        task_status = self.get_task_status(task_id=task_id)
        task_status.status = "pause"

    async def delete_model_download_task(self, task_id: str):
        task_status = self.get_task_status(task_id)
        is_running = task_status.status == "doing"
        task_status.status = "waiting"
        await utils.send_json("delete_download_task", task_id)

        # Pause the task
        if is_running:
            task_status.status = "pause"
            time.sleep(1)

        download_dir = utils.get_download_path()
        task_file_list = os.listdir(download_dir)
        for task_file in task_file_list:
            task_file_target = os.path.splitext(task_file)[0]
            if task_file_target == task_id:
                self.delete_task_status(task_id)
                os.remove(utils.join_path(download_dir, task_file))

        await utils.send_json("delete_download_task", task_id)

    async def download_model(self, task_id: str, request):
        main_loop = asyncio.get_running_loop()

        def download_task_sync(task_id: str):
            def run_on_main_loop(coro):
                asyncio.run_coroutine_threadsafe(coro, main_loop)

            def report_progress_sync(task_status: TaskStatus):
                run_on_main_loop(utils.send_json("update_download_task", task_status.to_dict()))

            try:
                task_status = self.get_task_status(task_id)
            except Exception:
                return

            task_status.status = "doing"
            run_on_main_loop(utils.send_json("update_download_task", task_status.to_dict()))

            try:
                headers = {"User-Agent": config.user_agent}
                download_platform = task_status.platform
                if download_platform == "civitai":
                    api_key = self.api_key.get_value("civitai")
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                elif download_platform == "huggingface":
                    api_key = self.api_key.get_value("huggingface")
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                progress_interval = 1.0
                self.download_model_file(
                    task_id=task_id,
                    headers=headers,
                    progress_callback=report_progress_sync,
                    interval=progress_interval,
                    run_on_main_loop_func=run_on_main_loop,
                )
            except Exception as e:
                task_status.status = "pause"
                task_status.error = str(e)
                run_on_main_loop(utils.send_json("update_download_task", task_status.to_dict()))
                task_status.error = None
                utils.print_error(str(e))

        try:
            status = self.download_thread_pool.submit(download_task_sync, task_id)
            if status == "Waiting":
                task_status = self.get_task_status(task_id)
                task_status.status = "waiting"
                await utils.send_json("update_download_task", task_status.to_dict())
        except Exception as e:
            try:
                task_status = self.get_task_status(task_id)
                task_status.status = "pause"
                task_status.error = str(e)
                await utils.send_json("update_download_task", task_status.to_dict())
                task_status.error = None
            except Exception as final_e:
                utils.print_error(f"Error in download_model exception handler: {final_e}")
            utils.print_error(str(e))

    def download_model_file(
        self,
        task_id: str,
        headers: dict,
        progress_callback: Callable[[TaskStatus], None],
        interval: float = 1.0,
        run_on_main_loop_func: Callable[[Awaitable], None] = None,
    ):
        download_path = utils.get_download_path()
        task_status = self.get_task_status(task_id)
        task_content = self.get_task_content(task_id)
        download_tmp_file = utils.join_path(download_path, f"{task_id}.download")

        def download_complete():
            model_type = task_content.type
            path_index = task_content.pathIndex
            fullname = task_content.fullname
            description = task_content.description
            description_file = utils.join_path(download_path, f"{task_id}.md")
            with open(description_file, "w", encoding="utf-8", newline="") as f:
                f.write(description)
            model_path = utils.get_full_path(model_type, path_index, fullname)
            utils.rename_model(download_tmp_file, model_path)
            time.sleep(1)
            task_file = utils.join_path(download_path, f"{task_id}.task")
            os.remove(task_file)
            if run_on_main_loop_func:
                run_on_main_loop_func(utils.send_json("complete_download_task", task_id))

        last_update_time = time.time()
        last_downloaded_size = 0
        downloaded_size = 0
        if os.path.isfile(download_tmp_file):
            downloaded_size = os.path.getsize(download_tmp_file)
            headers["Range"] = f"bytes={downloaded_size}-"
        
        total_size = task_content.sizeBytes

        def update_progress():
            nonlocal last_update_time, last_downloaded_size
            current_time = time.time()
            delta_time = current_time - last_update_time
            if delta_time > 0:
                bps = (downloaded_size - last_downloaded_size) / delta_time
            else:
                bps = 0
            
            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
            task_status.downloadedSize = downloaded_size
            task_status.progress = progress
            task_status.bps = bps
            progress_callback(task_status)
            last_update_time = current_time
            last_downloaded_size = downloaded_size

        if total_size > 0 and downloaded_size == total_size:
            download_complete()
            return

        last_downloaded_size = downloaded_size
        model_url = task_content.downloadUrl
        if not model_url:
            raise RuntimeError("No downloadUrl found")

        response = requests.get(
            url=model_url,
            headers=headers,
            stream=True,
            allow_redirects=True,
        )

        if response.status_code not in (200, 206):
            raise RuntimeError(f"Failed to download {task_content.fullname}, status code: {response.status_code}")

        content_type = response.headers.get("content-type")
        if content_type and content_type.startswith("text/html"):
            raise RuntimeError(f"{task_content.fullname} needs to be logged in to download. Please set the API-Key first.")

        if response.status_code == 200:
            response_total_size = float(response.headers.get("content-length", 0))
            if total_size == 0 or total_size != response_total_size:
                total_size = response_total_size
        elif response.status_code == 206:
            content_range = response.headers.get("content-range")
            if content_range:
                response_total_size = float(content_range.split("/")[-1])
                if total_size == 0 or total_size != response_total_size:
                    total_size = response_total_size

        if task_content.sizeBytes != total_size:
            task_content.sizeBytes = total_size
            task_status.totalSize = total_size
            self.set_task_content(task_id, task_content)
            if run_on_main_loop_func:
                run_on_main_loop_func(utils.send_json("update_download_task", task_status.to_dict()))

        with open(download_tmp_file, "ab") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if task_status.status == "pause":
                    break
                f.write(chunk)
                downloaded_size += len(chunk)
                if time.time() - last_update_time >= interval:
                    update_progress()

        update_progress()

        if total_size > 0 and downloaded_size == total_size:
            download_complete()
        else:
            task_status.status = "pause"
            if run_on_main_loop_func:
                run_on_main_loop_func(utils.send_json("update_download_task", task_status.to_dict()))
