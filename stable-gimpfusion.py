# vim: set noai ts=4 sw=4 expandtab
#!/usr/bin/python

# Stable Gimpfusion 
# v1.0.8
# Thin API client for Automatic1111's StableDiffusion API
# https://github.com/AUTOMATIC1111/stable-diffusion-webui

import base64
import json
import os
import random
import tempfile
import logging
import urllib
import urllib2
import gimp
import gimpenums
from gimpfu import *
from gimpshelf import shelf

DEBUG = False
VERSION = 8
PLUGIN_VERSION_URL = "https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/main/version.json"
MAX_BATCH_SIZE = 20

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.WARNING)

STABLE_GIMPFUSION_DEFAULT_SETTINGS = {
        "sampler_name": "Euler a",
        "denoising_strength": 0.8,
        "cfg_scale": 7.5,
        "steps": 50,
        "prompt": "",
        "negative_prompt": "",
        "batch_size": 1,
        "mask_blur": 4,
        "seed": -1,
        "api_base": "http://127.0.0.1:7860",
        "model": ""
        }

RESIZE_MODES = {
        "Just Resize": 0,
        "Crop And Resize": 1,
        "Resize And Fill": 2,
        "Just Resize (Latent Upscale)": 3
        }

SAMPLERS = [
      "Euler a",
      "Euler",
      "LMS",
      "Heun",
      "DPM2",
      "DPM2 a",
      "DPM++ 2S a",
      "DPM++ 2M",
      "DPM++ SDE",
      "DPM fast",
      "DPM adaptive",
      "LMS Karras",
      "DPM2 Karras",
      "DPM2 a Karras",
      "DPM++ 2S a Karras",
      "DPM++ 2M Karras",
      "DPM++ SDE Karras",
      "DDIM"
    ]

CONTROLNET_RESIZE_MODES = [
        "Just Resize",
        "Scale to Fit (Inner Fit)",
        "Envelope (Outer Fit)",
        ]

CONTROLNET_MODULES = [
        "none",
        "canny",
        "depth",
        "depth_leres",
        "hed",
        "mlsd",
        "normal_map",
        "openpose",
        "openpose_hand",
        "clip_vision",
        "color",
        "pidinet",
        "scribble",
        "fake_scribble",
        "segmentation",
        "binary"
        ]

CONTROLNET_DEFAULT_SETTINGS = {
      "input_image": "",
      "mask": "",
      "module": "none",
      "model": "none",
      "weight": 1.0,
      "resize_mode": "Scale to Fit (Inner Fit)",
      "lowvram": False,
      "processor_res": 64,
      "threshold_a": 64,
      "threshold_b": 64,
      "guidance": 1.0,
      "guidance_start": 0.0,
      "guidance_end": 1.0,
      "guessmode": True
    }

GENERATION_MESSAGES = [
        "Making happy little pixels...",
        "Fetching pixels from a digital art museum...",
        "Waiting for bot-painters to finish...",
        "Waiting for the prompt to bake...",
        "Fetching random pixels from the internet",
        "Taking a random screenshot from an AI dream",
        "Throwing pixels at screen and seeing what sticks",
        "Converting random internet comment to RGB values",
        "Computer make pretty picture, you happy.",
        "Computer is hand-painting pixels...",
        "Turning the Gimp knob up to 11...",
        "Pixelated dreams come true, thanks to AI.",
        "AI is doing its magic...",
        "Pocket Picasso is speed-painting...",
        "Instant Rembrandt! Well, relatively instant...",
        "Doodle buddy is doing its thing...",
        "Waiting for the digital paint to dry..."
        ]

STABLE_GIMPFUSION_DATA = {
        "models": [],
        "cn_models": [],
        "api_base": STABLE_GIMPFUSION_DEFAULT_SETTINGS["api_base"],
        "sd_model_checkpoint": None,
        "is_server_running": False
        }

def roundToMultiple(value, multiple):
    return multiple * round(float(value)/multiple)

def deunicodeDict(data):
    """Recursively converts dictionary keys to strings."""
    if isinstance(data, unicode):
        return str(data)
    if not isinstance(data, dict):
        return data
    return dict((str(k), deunicodeDict(v)) 
        for k, v in data.items())

class ApiClient():
    """ Simple API client used to interface with StableDiffusion JSON endpoints """
    def __init__(self, base_url):
        self.setBaseUrl(base_url)

    def setBaseUrl(self, base_url):
        self.base_url = base_url

    def post(self, endpoint, data={}, params={}, headers=None):
        try:
            url = self.base_url + endpoint + "?" + urllib.urlencode(params)
            logging.debug("POST "+url)
            data = json.dumps(data)

            logging.debug('post data %s', data)

            headers = headers or {"Content-Type": "application/json", "Accept": "application/json"}
            request = urllib2.Request(url=url, data=data, headers=headers)
            response = urllib2.urlopen(request)
            data = response.read()
            data = json.loads(data)

            logging.debug('response: %s', data)
            return data
        except Exception as ex:
            logging.exception("ERROR: ApiClient.post")

    def get(self, endpoint, params={}, headers=None):
        try:
            url = self.base_url + endpoint + "?" + urllib.urlencode(params)
            print("GET "+url)
            headers = headers or {"Content-Type": "application/json", "Accept": "application/json"}
            request = urllib2.Request(url=url, headers=headers)
            response = urllib2.urlopen(request)
            data = response.read()
            data = json.loads(data)
            return data
        except Exception as ex:
            logging.exception("ERROR: ApiClient.get")


class Rect():
    def __init__(self, width, height=None):
        global DEBUG
        if isinstance(width, Rect):
            self.width = width.width
            self.height = width.height
            self.orig_scale = width.orig_scale
        elif isinstance(width, list):
            self.width = width[0]
            self.height = width[1]
            self.orig_scale = float((len(width) == 3 and width[2]) or 1.0)
        elif isinstance(width, tuple):
            self.width = width[0]
            self.height = width[1]
            self.orig_scale = float((len(width) == 3 and width[2]) or 1.0)
        else:
            if type(width) == int or type(width) == float:
                self.width = width
            else:
                if DEBUG:
                    raise Exception("Width must be of type int or float")
            if type(height) == int or type(height) == float:
                self.height = height
            else:
                if DEBUG:
                    raise Exception("Height must be of type int or float")
            self.orig_scale = 1.0
        self.aspect = float(self.width) / self.height

    def scaleTo(self, new_scale):
        new_width = roundToMultiple(float(self.width) * new_scale, 64)
        new_height = roundToMultiple(float(self.height) * new_scale, 64)
        new_scale = new_width / self.width
        self.width = new_width
        self.height = new_height
        self.orig_scale = self.orig_scale * new_scale
        return self

    def fitIntoRect(self, target_rect, force = False):
        if (self.width < target_rect.width) and (self.height < target_rect.height) and not force:
            # nothing to do, rect already fits inside target
            return
        if self.aspect > target_rect.aspect:
            scale = float(target_rect.width) / self.width
        else:
            scale = float(target_rect.height) / self.height
        return self.scaleTo(scale)

    def coverRect(self, target_rect, force = False):
        if (self.width > target_rect.width) and (self.height > target_rect.height) and not force:
            # nothing to do, rect already covers target
            return
        if self.aspect > target_rect.aspect:
            scale = float(target_rect.height) / self.height
        else:
            scale = float(target_rect.width) / self.width
        return self.scaleTo(scale)

    def fitBetween(self, min_rect, max_rect):
        self.coverRect(Rect(min_rect))
        self.fitIntoRect(Rect(max_rect))
        self.scaleTo(1.0)
        return self

    def __str__ (self):
        return 'Rect('+", ".join(map(str,[self.width, self.height, self.aspect]))+')'


class StableDiffusionOptions():
    """ Helper class to interface with the StableDiffusion options endpoint.
        Used to change the checkpoint model.
    """
    def __init__(self, api):
        self.name = "stable_diffusion_options"
        self.options = {}
        self.api = api
        try:
            self.load()
        except Exception as ex:
            global DEBUG
            if DEBUG:
                raise ex

    def load(self):
        try:
            self.options = self.api.get("/sdapi/v1/options") or {}
        except Exception as ex:
            logging.exception("ERROR: StableDiffusionOptions.load")

    def get(self, name, default_value=None):
        if name in self.options:
            return self.options[name]
        return default_value

    def set(self, name, value):
        try:
            data = {}
            data[name] = value
            self.api.post("/sdapi/v1/options", data=data)
            self.options[name] = value
        except Exception as ex:
            logging.exception("ERROR: StableDiffusionOptions.set")


class DynamicDropdownData():
    """ Get the StableDiffusion data needed for dynamic PF_OPTION lists """
    def __init__(self):
        self.name = "stable_diffusion_data"
        self.settings = STABLE_GIMPFUSION_DATA.copy()
        try:
            self.fetch()
        except Exception as ex:
            pass

    def fetch(self):
        api = ApiClient(self.settings["api_base"])
        try:
            self.settings["models"] = map(lambda data: data["title"], api.get("/sdapi/v1/sd-models") or [])
            self.settings["cn_models"] = api.get("/controlnet/model_list")["model_list"] or []
            options = api.get("/sdapi/v1/options")
            self.settings["sd_model_checkpoint"] = options["sd_model_checkpoint"] or None
            self.settings["is_server_running"] = True
        except Exception as ex:
            logging.exception("ERROR: DynamicDropdownData.fetch")
            self.settings["is_server_running"] = False

    def get(self, name, default_value=None):
        if name in self.settings:
            return self.settings[name]
        return default_value

sd_data = DynamicDropdownData()
models = sd_data.get("models", [])
sd_model_checkpoint = sd_data.get("sd_model_checkpoint")
is_server_running = sd_data.get("is_server_running")

class StableGimpfusionPlugin():
    def __init__(self, image):
        self.name = "stable_gimpfusion"
        self.image = image
        self.loadSettings()

        global is_server_running
        if not is_server_running:
            gimp.pdb.gimp_message("It seems that StableDiffusion is not runing on "+self.settings["api_base"])

        try:
            self.api = ApiClient(self.settings["api_base"])
            self.files = TempFiles()
            self.options = StableDiffusionOptions(self.api)
        except Exception as e:
            logging.exception("ERROR: StableGimpfusionPlugin.__init__")

    def loadSettings(self):
        parasite = self.image.parasite_find(self.name)
        if not parasite:
            self.settings = deunicodeDict(STABLE_GIMPFUSION_DEFAULT_SETTINGS)
        else:
            self.settings = deunicodeDict(json.loads(parasite.data))

    def saveSettings(self, settings):
        parasite = gimp.Parasite(self.name, gimpenums.PARASITE_PERSISTENT, deunicodeDict(json.dumps(settings)))
        self.image.parasite_attach(parasite)

    def checkUpdate(self):
        try:
            gimp.get_data("update_checked")
            updateChecked = True
        except Exception as ex:
            updateChecked = False

        if updateChecked is False:
            try:
                response = urllib2.urlopen(PLUGIN_VERSION_URL)
                data = response.read()
                data = json.loads(data)
                gimp.set_data("update_checked", "1")

                if VERSION < int(data["version"]):
                    gimp.pdb.gimp_message(data["message"])
            except Exception as ex:
                ex = ex

    def getLayerAsBase64(self, layer, rect = None):
        # store active_layer
        active_layer = layer.image.active_layer
        copy = Layer(layer).copy().insert()
        if rect is not None:
            copy.resize(rect.width, rect.height)
        result = copy.toBase64();
        copy.remove()
        # restore active_layer
        gimp.pdb.gimp_image_set_active_layer(active_layer.image, active_layer)
        return result

    def getActiveLayerAsBase64(self, rect = None):
        return self.getLayerAsBase64(self.image.active_layer, rect)

    def getLayerMaskAsBase64(self, layer, rect = None):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(layer.image)
        if non_empty:
            # selection to base64

            # store active_layer
            active_layer = layer.image.active_layer

            # selection to file
            #disable=pdb.gimp_image_undo_disable(layer.image)
            tmp_layer = Layer.create(layer.image, "mask", layer.image.width, layer.image.height, RGBA_IMAGE, 100, NORMAL_MODE)
            tmp_layer.addSelectionAsMask().insert()

            if rect is not None:
                tmp_layer.resize(rect.width, rect.height)

            result = tmp_layer.maskToBase64()
            tmp_layer.remove()
            #enable = pdb.gimp_image_undo_enable(layer.image)

            # restore active_layer
            gimp.pdb.gimp_image_set_active_layer(active_layer.image, active_layer)

            return result
        elif layer.mask:
            # mask to file
            tmp_layer = Layer(layer).scale(scale)
            if rect is not None:
                tmp_layer.resize(rect.width, rect.height)
            return tmp_layer.maskToBase64()
        else:
            return ""

    def getActiveMaskAsBase64(self, rect = None):
        return self.getLayerMaskAsBase64(self.image.active_layer, rect)

    def getSelectionBounds(self):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(self.image)
        if non_empty:
            return x1, y1, x2-x1, y2-y1
        return 0, 0, self.image.width, self.image.height

    def cleanup(self):
        self.files.removeAll()
        self.checkUpdate()

    def getControlNetParams(self, cn_layer):
        if cn_layer:
            layer = Layer(cn_layer)
            data = layer.loadData(CONTROLNET_DEFAULT_SETTINGS)
            # ControlNet image size need to be in multiples of 64
            layer64 = layer.copy().insert().resizeToMultipleOf(64)
            data.update({"input_image": layer64.toBase64()})
            if cn_layer.mask:
                data.update({"mask": layer64.maskToBase64()})
            layer64.remove()
            return data
        return None

    def imageToImage(self, *args):
        resize_mode, prompt, negative_prompt, seed, batch_size, steps, mask_blur, cfg_scale, denoising_strength, sampler_index, cn_enabled, cn_layer, cn_skip_annotator_layers = args
        image = self.image
        settings = self.settings

        rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        data = {
            "resize_mode": resize_mode,
            "init_images": [self.getActiveLayerAsBase64(rect)],

            "prompt": (prompt + " " + settings["prompt"]).strip(),
            "negative_prompt": (negative_prompt + " " +  settings["negative_prompt"]).strip(),
            "denoising_strength": float(denoising_strength),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "width": int(width),
            "height": int(height),
            "sampler_index": SAMPLERS[sampler_index],
            "batch_size": min(MAX_BATCH_SIZE, max(1, batch_size)),
            "seed": seed or -1
        }

        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))

            if cn_enabled:
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/img2img", data)
                # last image is the controlnet mask, ignore it
                #if len(response["images"]) > 1:
                #    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/img2img", data)

            ResponseLayers(image, response, {"skip_annotator_layers": cn_skip_annotator_layers}).resize(image.width, image.height)

        except Exception as ex:
            logging.exception("ERROR: StableGimpfusionPlugin.imageToImage")
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

    def inpainting(self, *args):
        resize_mode, prompt, negative_prompt, seed, batch_size, steps, mask_blur, cfg_scale, denoising_strength, sampler_index, cn_enabled, cn_layer, cn_skip_annotator_layers = args
        image = self.image
        settings = self.settings

        rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        init_images = [self.getActiveLayerAsBase64(rect)]
        mask = self.getActiveMaskAsBase64(rect)
        if mask == "":
            logging.exception("ERROR: StableGimpfusionPlugin.inpainting")
            raise Exception("Inpainting must use either a selection or layer mask")

        data = {
            "mask": mask,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 10,
            "inpainting_mask_invert": 0,

            "resize_mode": resize_mode,
            "init_images": init_images,

            "prompt": (prompt + " " + settings["prompt"]).strip(),
            "negative_prompt": (negative_prompt + " " +  settings["negative_prompt"]).strip(),
            "denoising_strength": float(denoising_strength),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "width": width,
            "height": height,
            "sampler_index": SAMPLERS[sampler_index],
            "batch_size": min(MAX_BATCH_SIZE, max(1, batch_size)),
            "seed": seed or -1
        }

        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))

            if cn_enabled:
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/img2img", data)
                # last image is the controlnet mask, ignore it
                #if len(response["images"]) > 1:
                #    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/img2img", data)

            ResponseLayers(image, response, {"skip_annotator_layers": cn_skip_annotator_layers}).resize(image.width, image.height)

        except Exception as ex:
            logging.exception("ERROR: StableGimpfusionPlugin.inpainting")
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

    def textToImage(self, *args):
        prompt, negative_prompt, seed, batch_size, steps, mask_blur, cfg_scale, denoising_strength, sampler_index, cn_enabled, cn_layer, cn_skip_annotator_layers = args
        image = self.image
        settings = self.settings

        x, y, width, height = self.getSelectionBounds()
        rect = Rect(width, height).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        data = {
            "prompt": (prompt + " " + settings["prompt"]).strip(),
            "negative_prompt": (negative_prompt + " " +  settings["negative_prompt"]).strip(),
            "cfg_scale": float(cfg_scale),
            "denoising_strength": float(denoising_strength),
            "steps": int(steps),
            "width": int(width),
            "height": int(height),
            "sampler_index": SAMPLERS[sampler_index],
            "batch_size": min(MAX_BATCH_SIZE, max(1, batch_size)),
            "seed": seed or -1
        }

        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))

            if cn_enabled:
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/txt2img", data)
                # last image is the controlnet mask, ignore it
                #if len(response["images"]) > 1:
                #    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/txt2img", data)

            ResponseLayers(image, response, {"skip_annotator_layers": cn_skip_annotator_layers}).scale(1.0/rect.orig_scale).translate((x, y)).addSelectionAsMask()

        except Exception as ex:
            logging.exception("ERROR: StableGimpfusionPlugin.textToImage")
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

    def showLayerInfo(self, *args):
        """ Show any layer info associated with the active layer """

        data = LayerData(self.image.active_layer).data
        gimp.pdb.gimp_message("This layer has the following data associated with it\n" + json.dumps(data, sort_keys=True, indent=4))


    def saveControlLayer(self, module, model, weight, resize_mode, lowvram, guessmode, guidance_start, guidance_end, guidance, processor_res, threshold_a, threshold_b):
        """ Take the form params and save them to the layer as gimp.Parasite """
        cn_models = sd_data.get("cn_models", [])
        settings = {
            "module": CONTROLNET_MODULES[module],
            "model": cn_models[model],
            "weight": weight,
            "resize_mode": CONTROLNET_RESIZE_MODES[resize_mode],
            "lowvram": lowvram,
            "guessmode": guessmode,
            "guidance_start": guidance_start,
            "guidance_end": guidance_end,
            "guidance": guidance,
            "processor_res": processor_res,
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
        }
        active_layer = self.image.active_layer
        cnlayer = LayerData(active_layer, CONTROLNET_DEFAULT_SETTINGS)
        if not cnlayer.had_parasite:
            pdb.gimp_layer_set_name(active_layer, "ControlNet Layer")
        cnlayer.save(settings)

    def config(self, prompt, negative_prompt, url):
        #model = sd_data.get("models")[model]
        settings = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "api_base": url,
        }
        self.saveSettings(settings)

    def changeModel(self, model):
        if self.settings["model"] != model:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text("Changing model...")
            self.options.set("sd_model_checkpoint", model)
            gimp.pdb.gimp_progress_end()


class TempFiles(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(TempFiles, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.files = []

    def get(self, filename):
        self.files.append(filename)
        return r"{}".format(os.path.join(tempfile.gettempdir(), filename))

    def removeAll(self):
        try:
            unique_list = (list(set(self.files)))
            for tmpfile in unique_list:
                if os.path.exists(tmpfile):
                    os.remove(tmpfile)
        except Exception as ex:
            ex = ex


class LayerData():
    def __init__(self, layer, defaults = {}):
        self.name = 'gimpfusion'
        self.layer = layer
        self.image = layer.image
        self.defaults = defaults;
        self.had_parasite = False
        self.load()

    def load(self):
        parasite = self.layer.parasite_find(self.name)
        if not parasite:
            self.data = self.defaults.copy()
        else:
            self.had_parasite = True
            self.data = json.loads(parasite.data)
        self.data = deunicodeDict(self.data)
        return self.data

    def save(self, data):
        parasite = gimp.Parasite(self.name, gimpenums.PARASITE_PERSISTENT, deunicodeDict(json.dumps(data)))
        self.layer.parasite_attach(parasite)


layer_id = 1
class Layer():
    def __init__(self, layer = None):
        global layer_id
        self.id = layer_id 
        layer_id += 1
        if layer is not None:
            self.layer = layer
            self.image = layer.image

    @staticmethod
    def create(image, name, width, height, image_type, opacity, mode):
        layer = gimp.Layer(image, name, width, height, image_type, opacity, mode)
        return Layer(layer)

    @staticmethod
    def fromBase64(img, base64Data):
        filepath = TempFiles().get("generated.png")
        imageFile = open(filepath, "wb+")
        imageFile.write(base64.b64decode(base64Data))
        imageFile.close()
        layer = gimp.pdb.gimp_file_load_layer(img, filepath)
        return Layer(layer)


    def rename(self, name):
        pdb.gimp_layer_set_name(self.layer, name)
        return self;

    def saveData(self, data):
        LayerData(self.layer).save(data)
        return self

    def loadData(self, default_data):
        return LayerData(self.layer, default_data).data.copy()

    def copy(self):
        copy = gimp.pdb.gimp_layer_copy(self.layer, True)
        return Layer(copy)

    def scale(self, new_scale=1.0):
        if new_scale != 1.0:
            gimp.pdb.gimp_layer_scale(self.layer, int(new_scale * self.layer.width), int(new_scale * self.layer.height), False)
        return self

    def resize(self, width, height):
        logging.info("Resizing to %dx%d", width, height)
        gimp.pdb.gimp_layer_scale(self.layer, width, height, False)

    def resizeToMultipleOf(self, multiple):
        gimp.pdb.gimp_layer_scale(self.layer, roundToMultiple(self.layer.width, multiple), roundToMultiple(self.layer.height, multiple), False)
        return self

    def translate(self, offset=None):
        if offset is not None:
            gimp.pdb.gimp_layer_set_offsets(self.layer, offset[0], offset[1])
        return self

    def insert(self):
        gimp.pdb.gimp_image_insert_layer(self.image, self.layer, None, -1)
        return self

    def insertTo(self, image=None):
        image = image or self.image
        gimp.pdb.gimp_image_insert_layer(image, self.layer, None, -1)
        return self

    def addSelectionAsMask(self):
        mask = self.layer.create_mask(ADD_SELECTION_MASK)
        self.layer.add_mask(mask)
        return self

    def saveMaskAs(self, filepath):
        gimp.pdb.file_png_save(self.image, self.layer.mask, filepath, filepath, False, 9, True, True, True, True, True)
        return self

    def saveAs(self, filepath):
        gimp.pdb.file_png_save(self.image, self.layer, filepath, filepath, False, 9, True, True, True, True, True)
        return self

    def maskToBase64(self):
        filepath = TempFiles().get("mask"+str(self.id)+".png")
        self.saveMaskAs(filepath)
        file = open(filepath, "rb")
        return base64.b64encode(file.read())

    def toBase64(self):
        filepath = TempFiles().get("layer"+str(self.id)+".png")
        self.saveAs(filepath)
        file = open(filepath, "rb")
        return base64.b64encode(file.read())

    def remove(self):
        gimp.pdb.gimp_image_remove_layer(self.layer.image, self.layer)
        return self


class ResponseLayers():
    def __init__(self, img, response, options = {}):
        self.image = img
        color = gimp.pdb.gimp_context_get_foreground()
        gimp.pdb.gimp_context_set_foreground((0, 0, 0))

        layers = []
        try:
            info = json.loads(response["info"])
            infotexts = info["infotexts"]
            seeds = info["all_seeds"]
            index = 0
            logging.debug(infotexts)
            logging.debug(seeds)
            total_images = len(seeds)
            for image in response["images"]:
                if index < total_images:
                    layer_data = {"info": infotexts[index], "seed": seeds[index]}
                    layer = Layer.fromBase64(img, image).rename("Generated Layer "+str(seeds[index])).saveData(layer_data).insertTo(img)
                else:
                    # annotator layers
                    if "skip_annotator_layers" in options and not options["skip_annotator_layers"]:
                        layer = Layer.fromBase64(img, image).rename("Annotator Layer").insertTo(img)
                layers.append(layer.layer)
                index += 1
        except Exception as e:
            logging.exception("ResponseLayers")

        gimp.pdb.gimp_context_set_foreground(color)
        self.layers = layers

    def scale(self, new_scale=1.0):
        if new_scale != 1.0:
            for layer in self.layers:
                Layer(layer).scale(new_scale)
        return self

    def resize(self, width, height):
        for layer in self.layers:
            Layer(layer).resize(width, height)
        return self

    def translate(self, offset=None):
        if offset is not None:
            for layer in self.layers:
                Layer(layer).translate(offset)
        return self

    def insertTo(self, image=None):
        image = image or self.image
        for layer in self.layers:
            Layer(layer).insertTo(image)
        return self

    def addSelectionAsMask(self):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(self.image)
        if not non_empty:
            return
        for layer in self.layers:
            Layer(layer).addSelectionAsMask()
        return self

def handleConfig(image, drawable, *args):
    StableGimpfusionPlugin(image).config(*args)

def handleChangeModel(image, drawable, *args):
    StableGimpfusionPlugin(image).changeModel(*args)

def handleImageToImage(image, drawable, *args):
    StableGimpfusionPlugin(image).imageToImage(*args)

def handleImageToImageFromLayersContext(image, drawable, *args):
    StableGimpfusionPlugin(image).imageToImage(*args)

def handleInpainting(image, drawable, *args):
    StableGimpfusionPlugin(image).inpainting(*args)

def handleInpaintingFromLayersContext(image, drawable, *args):
    StableGimpfusionPlugin(image).inpainting(*args)

def handleTextToImage(image, drawable, *args):
    StableGimpfusionPlugin(image).textToImage(*args)

def handleTextToImageFromLayersContext(image, drawable, *args):
    StableGimpfusionPlugin(image).textToImage(*args)

def handleControlNetLayerConfig(image, drawable, *args):
    StableGimpfusionPlugin(image).saveControlLayer(*args)

def handleControlNetLayerConfigFromLayersContext(image, drawable, *args):
    StableGimpfusionPlugin(image).saveControlLayer(*args)

def handleShowLayerInfo(image, drawable, *args):
    StableGimpfusionPlugin(image).showLayerInfo(*args)

def handleShowLayerInfoContext(image, drawable, *args):
    StableGimpfusionPlugin(image).showLayerInfo(*args)

PLUGIN_FIELDS_IMAGE = [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
        ]

PLUGIN_FIELDS_LAYERS = [
        (PF_IMAGE, "image", "Image", None),
        (PF_LAYER, "layer", "Layer", None),
        ]

PLUGIN_FIELDS_COMMON = [
        (PF_TEXT, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
        (PF_TEXT, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"]),
        (PF_INT32, "seed", "Seed", STABLE_GIMPFUSION_DEFAULT_SETTINGS["seed"]),
        (PF_SLIDER, "batch_size", "Batch count", STABLE_GIMPFUSION_DEFAULT_SETTINGS["batch_size"], (1, 20, 1.0)),
        (PF_SLIDER, "steps", "Steps", STABLE_GIMPFUSION_DEFAULT_SETTINGS["steps"], (10, 150, 1.0)),
        (PF_SLIDER, "mask_blur", "Mask Blur", STABLE_GIMPFUSION_DEFAULT_SETTINGS["mask_blur"], (1, 10, 1.0)),
        (PF_SLIDER, "cfg_scale", "CFG Scale", STABLE_GIMPFUSION_DEFAULT_SETTINGS["cfg_scale"], (0, 20, 0.5)),
        (PF_SLIDER, "denoising_strength", "Denoising Strength", STABLE_GIMPFUSION_DEFAULT_SETTINGS["denoising_strength"], (0.0, 1.0, 0.1)),
        (PF_OPTION, "sampler_index", "Sampler", SAMPLERS.index(STABLE_GIMPFUSION_DEFAULT_SETTINGS["sampler_name"]), SAMPLERS),
        ]

PLUGIN_FIELDS_CONTROLNET_OPTIONS = [
        (PF_TOGGLE, "cn_enabled", "Enable ControlNet", False),
        (PF_LAYER, "cn_layer", "ControlNet Layer", None),
        (PF_TOGGLE, "cn_skip_annotator_layers", "Skip annotator layers", True),
        ]

PLUGIN_FIELDS_CONFIG = [
    (PF_STRING, "prompt", "Prompt Suffix", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
    (PF_STRING, "negative_prompt", "Negative Prompt Suffix", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"]),
    (PF_STRING, "api_base", "Backend API URL base", STABLE_GIMPFUSION_DEFAULT_SETTINGS["api_base"]),
    ]

if sd_model_checkpoint is not None:
    PLUGIN_FIELDS_CHECKPOINT = [
        (PF_OPTION, "model", "Model", models.index(sd_model_checkpoint), models)
        ]
else:
    PLUGIN_FIELDS_CHECKPOINT = []


PLUGIN_FIELDS_RESIZE_MODE = [(PF_OPTION, "resize_mode", "Resize Mode", 0, tuple(RESIZE_MODES.keys()))]
PLUGIN_FIELDS_TXT2IMG = [] + PLUGIN_FIELDS_COMMON + PLUGIN_FIELDS_CONTROLNET_OPTIONS
PLUGIN_FIELDS_IMG2IMG = [] + PLUGIN_FIELDS_RESIZE_MODE + PLUGIN_FIELDS_TXT2IMG

PLUGIN_FIELDS_CONTROLNET = [] + [
        (PF_OPTION, "module", "Module", 0, CONTROLNET_MODULES),
        (PF_OPTION, "model", "Model", 0, sd_data.get("cn_models", ["none"])),
        (PF_SLIDER, "weight",  "Weight", 1, (0, 2, 0.05)),
        (PF_OPTION, "resize_mode", "Resize Mode", 1, CONTROLNET_RESIZE_MODES),
        (PF_BOOL, "lowvram", "Low VRAM", False),
        (PF_BOOL, "guessmode", "Guess Mode", False),
        (PF_SLIDER, "guidance_start",  "Guidance Start (T)", 0, (0, 1, 0.01)),
        (PF_SLIDER, "guidance_end",  "Guidance End (T)", 1, (0, 1, 0.01)),
        (PF_SLIDER, "guidance",  "Guidance", 1, (0, 1, 0.01)),
        (PF_SLIDER, "processor_res",  "Processor Resolution", 512, (64, 2048, 1)),
        (PF_SLIDER, "threshold_a",  "Threshold A", 64, (100, 2048, 1)),
        (PF_SLIDER, "threshold_b",  "Threshold B", 64, (200, 2048, 1)),
        ]

register(
        "stable-gimpfusion-config",
        "This is where you configure params that are shared between all API requests",
        "Gimp Client for the StableDiffusion Automatic1111 API",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Config/Global",
        "*",
        PLUGIN_FIELDS_CONFIG,
        [],
        handleConfig
        )

register(
        "stable-gimpfusion-config-model",
        "Change the Checkpoint Model",
        "Change the Checkpoint Model",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Config/Change Model",
        "*",
        PLUGIN_FIELDS_CHECKPOINT,
        [],
        handleChangeModel
        )

register(
        "stable-gimpfusion-txt2img",
        "Text to image",
        "Text to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Text to image",
        "*",
        [] + PLUGIN_FIELDS_TXT2IMG,
        [],
        handleTextToImage
        )


register(
        "stable-gimpfusion-txt2img-context",
        "Text to image",
        "Text to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Layers>/GimpFusion/Text to image",
        "*",
        [] + PLUGIN_FIELDS_LAYERS + PLUGIN_FIELDS_TXT2IMG,
        [],
        handleTextToImageFromLayersContext
        )


register(
        "stable-gimpfusion-img2img",
        "Image to image",
        "Image to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Image to image",
        "*",
        [] + PLUGIN_FIELDS_IMG2IMG,
        [],
        handleImageToImage
        )

register(
        "stable-gimpfusion-img2img-context",
        "Image to image",
        "Image to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Layers>/GimpFusion/Image to image",
        "*",
        [] + PLUGIN_FIELDS_LAYERS + PLUGIN_FIELDS_IMG2IMG,
        [],
        handleImageToImageFromLayersContext
        )

register(
        "stable-gimpfusion-inpainting",
        "Inpainting",
        "Inpainting",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Inpainting",
        "*",
        [] + PLUGIN_FIELDS_IMG2IMG,
        [],
        handleInpainting
        )

register(
        "stable-gimpfusion-inpainting-context",
        "Inpainting",
        "Inpainting",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Layers>/GimpFusion/Inpainting",
        "*",
        [] + PLUGIN_FIELDS_LAYERS + PLUGIN_FIELDS_IMG2IMG,
        [],
        handleInpaintingFromLayersContext
        )

register(
        "stable-gimpfusion-config-controlnet-layer",
        "Convert current layer to ControlNet layer or edit ControlNet Layer's options",
        "ControlNet Layer",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Active layer as ControlNet",
        "*",
        [] + PLUGIN_FIELDS_CONTROLNET,
        [],
        handleControlNetLayerConfig
        )

register(
        "stable-gimpfusion-config-controlnet-layer-context",
        "Convert current layer to ControlNet layer or edit ControlNet Layer's options",
        "ControlNet Layer",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Layers>/GimpFusion/Use as ControlNet",
        "*",
        [] + PLUGIN_FIELDS_LAYERS + PLUGIN_FIELDS_CONTROLNET,
        [],
        handleControlNetLayerConfigFromLayersContext
        )

register(
        "stable-gimpfusion-layer-info",
        "Show stable gimpfusion info associated with this layer",
        "Layer Info",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/GimpFusion/Config/Layer Info",
        "*",
        [],
        [],
        handleShowLayerInfo
        )

register(
        "stable-gimpfusion-layer-info-context",
        "Show stable gimpfusion info associated with this layer",
        "Layer Info",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Layers>/GimpFusion/Layer Info",
        "*",
        [] + PLUGIN_FIELDS_LAYERS,
        [],
        handleShowLayerInfoContext
        )

main()
