# vim: set noai ts=4 sw=4 expandtab
#!/usr/bin/python

# Stable Gimpfusion 
# v1.0.1
# Thin API client for Automatic1111's StableDiffusion API
# https://github.com/AUTOMATIC1111/stable-diffusion-webui

import base64
import gimp
import gimpenums
import json
import math
import os
import random
import tempfile
import urllib
import urllib2
from gimpfu import *
from gimpshelf import shelf
from pprint import pprint

VERSION = 2
PLUGIN_VERSION_URL = "https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/main/version.json"

STABLE_GIMPFUSION_DEFAULT_SETTINGS = {
        "sampler_name": "Euler a",
        "denoising_strength": 0.8,
        "cfg_scale": 7.5,
        "steps": 50,
        "seed": -1,
        "prompt": "",
        "negative_prompt": "",
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
      "weight": 1,
      "resize_mode": "Scale to Fit (Inner Fit)",
      "lowvram": False,
      "processor_res": 64,
      "threshold_a": 64,
      "threshold_b": 64,
      "guidance": 1,
      "guidance_start": 0,
      "guidance_end": 1,
      "guessmode": True
    }

GENERATION_MESSAGES = [
        "Making happy little pixels...",
        "Fetching pixels from a digital art museum...",
        "Waiting for bot-painters to finish...",
        "Waiting for the prompt to bake...",
        "Fetching random pixels from the internet",
        "Taking a random screenshot from an AI dream",
        "Throwing pixels at screen and seeing what happens",
        "Converting random internet comment to RGB values",
        "Computer make pretty picture, you happy.",
        "Computer is hand-painting pixels...",
        "Injecting steroids to Gimp. (Legal ones).",
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
        "api_base": STABLE_GIMPFUSION_DEFAULT_SETTINGS["api_base"]
        }

def roundToMultiple(value, multiple):
    return multiple * math.floor(value/multiple)

def deepMapDict(d, cb):
    d_type = type(d)
    if d_type is dict:
        return dict((k, deepMapDict(v, cb)) for k, v in d.iteritems() if v and deepMapDict(v, cb))
    if d_type is list:
        return map(lambda k: deepMapDict(k, cb), d)
    else:
        return cb(d)

def pprintCompact(o):
    pprint(deepMapDict(o, lambda o: o if type(o) != str else o[0:10] + "..." if len(o) > 10 else o))


class ApiClient():
    """ Simple API client used to interface with StableDiffusion JSON endpoints """
    def __init__(self, base_url):
        self.setBaseUrl(base_url)

    def setBaseUrl(self, base_url):
        self.base_url = base_url

    def post(self, endpoint, data={}, params={}, headers=None):
        try:
            url = self.base_url + endpoint + "?" + urllib.urlencode(params)
            print("POST "+url)
            pprintCompact(data)
            data = json.dumps(data)
            headers = headers or {"Content-Type": "application/json", "Accept": "application/json"}
            request = urllib2.Request(url=url, data=data, headers=headers)
            response = urllib2.urlopen(request)
            data = response.read()
            data = json.loads(data)
            return data
        except Exception as ex:
            print("ERROR: ApiClient.post")
            raise ex

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
            print("ERROR: ApiClient.get")
            raise ex


class Rect():
    def __init__(self, width, height=None):
        if isinstance(width, Rect):
            self.width = width.width
            self.height = width.height
            self.scale = width.scale
        elif isinstance(width, list):
            self.width = width[0]
            self.height = width[1]
            self.scale = float((len(width) == 3 and width[2]) or 1.0)
        elif isinstance(width, tuple):
            self.width = width[0]
            self.height = width[1]
            self.scale = float((len(width) == 3 and width[2]) or 1.0)
        else:
            if type(width) == int or type(width) == float:
                self.width = width
            else:
                raise Exception("Width must be of type int or float")
            if type(height) == int or type(height) == float:
                self.height = height
            else:
                raise Exception("Height must be of type int or float")
            self.scale = 1.0

    def scale(self, new_scale):
        self.width = float(self.width) / self.scale * new_scale
        self.height = float(self.height) / self.scale * new_scale
        self.scale = new_scale
        return self

    def scaleRectTo(self, target_rect, maximum=True):
        target = Rect(target_rect)
        aspect_ratio = float(self.width) / self.height
        new_width = roundToMultiple(int(target.height * aspect_ratio), 64)
        new_height = roundToMultiple(int(target.width / aspect_ratio), 64)
        new_scale = float(new_width) / self.width
        if maximum or (new_width >= target.width):
            self.width = new_width or 1
            self.height = target.height
            self.scale = new_scale
        self.width = target.width
        self.height = new_height or 1
        self.scale = new_scale
        return self

    def fitIntoRect(self, target_rect):
        return self.scaleRectTo(target_rect, True)

    def coverRect(self, target_rect):
        return self.scaleRectTo(target_rect, False)

    def fitBetween(self, min_rect, max_rect):
        min_rect = Rect(min_rect)
        max_rect = Rect(max_rect)
        if self.width > max_rect.width or self.height > max_rect.height:
            self.fitIntoRect(max_rect)
        elif self.width < min_rect.width or self.height < min_rect.height:
            self.coverRect(min_rect)
        return self


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
            raise ex

    def load(self):
        try:
            self.options = self.api.get("/sdapi/v1/options") or {}
        except Exception as ex:
            print("ERROR: StableDiffusionOptions.load")
            raise ex

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
            print("ERROR: StableDiffusionOptions.set")
            raise ex


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
            self.settings["sd_model_checkpoint"] = options["sd_model_checkpoint"] or 0
        except Exception as ex:
            print("ERROR: DynamicDropdownData.fetch")
            raise ex

    def get(self, name, default_value=None):
        if name in self.settings:
            return self.settings[name]
        return default_value

sd_data = DynamicDropdownData()


class StableGimpfusionPlugin():
    def __init__(self, image):
        self.name = "stable_gimpfusion"
        self.image = image
        self.loadSettings()
        self.api = ApiClient(self.settings["api_base"])
        self.files = TempFiles()
        self.options = StableDiffusionOptions(self.api)

    def loadSettings(self):
        parasite = self.image.parasite_find(self.name)
        if not parasite:
            self.settings = STABLE_GIMPFUSION_DEFAULT_SETTINGS
        else:
            self.settings = json.loads(parasite.data)

    def saveSettings(self, settings):
        parasite = gimp.Parasite(self.name, gimpenums.PARASITE_UNDOABLE, json.dumps(settings))
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

    def drawableToFile(self, drawable, filepath):
        gimp.pdb.file_png_save(self.image, drawable, filepath, filepath, False, 9, True, True, True, True, True)

    def fileToBase64(self, filepath):
        file = open(filepath, "rb")
        return base64.b64encode(file.read())

    def getLayerAsBase64(self, layer):
        filepath = self.files.get('layer.png')
        self.drawableToFile(layer, filepath)
        return self.fileToBase64(filepath)

    def getActiveLayerAsBase64(self):
        return self.getLayerAsBase64(self.image.active_layer)

    def getLayerMaskAsBase64(self, layer):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(layer.image)
        if non_empty:
            # selection to file
            layer = gimp.Layer(layer.image, "mask", layer.image.width, layer.image.height, RGBA_IMAGE, 100, NORMAL_MODE)
            mask = layer.create_mask(ADD_SELECTION_MASK)
            layer.add_mask(mask)
            filepath = self.files.get('selection.png')
            self.drawableToFile(layer.mask, filepath)
            return self.fileToBase64(filepath)
        elif layer.mask:
            # mask to file
            filepath = self.files.get('mask.png')
            self.drawableToFile(layer.mask, filepath)
            return self.fileToBase64(filepath)
        else:
            return None

    def getActiveMaskAsBase64(self):
        return self.getLayerMaskAsBase64(self.image.active_layer)

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
            data = ControlNetLayerData(cn_layer).data.copy()
            #if cn_layer != self.image.active_layer:
            data.update({"input_image": self.getLayerAsBase64(cn_layer)})
            data.update({"mask":""})
            #mask = self.getLayerMaskAsBase64(cn_layer)
            #if mask is not None:
            #    data.update({"mask": mask})
            print("ControlNet")
            pprintCompact(data)

            return data
        return None

    def imageToImage(self, resize_mode, prompt, negative_prompt, cn_enabled, cn_layer):
        image = self.image
        settings = self.settings
        rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        data = {
            "prompt": prompt + " " + settings["prompt"],
            "negative_prompt": negative_prompt + " " +  settings["negative_prompt"],
            "denoising_strength": float(settings["denoising_strength"]),
            "resize_mode": resize_mode,
            "steps": int(settings["steps"]),
            "cfg_scale": float(settings["cfg_scale"]),
            "width": int(width),
            "height": int(height),
            "init_images": [self.getActiveLayerAsBase64()],
            "sampler_index": settings["sampler_name"],
            "seed": int(settings["seed"])
        }
        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
            if cn_enabled:
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/img2img", data)
                # last image is the controlnet mask, ignore it
                if len(response["images"]) > 1:
                    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/img2img", data)

            Layers(image, response["images"]).insertTo(image).scale(1.0/rect.scale)
        except Exception as ex:
            print("ERROR: StableGimpfusionPlugin.imageToImage")
            raise ex
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

    def inpainting(self, resize_mode, prompt, negative_prompt, cn_enabled, cn_layer):

        image = self.image
        settings = self.settings
        rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        mask = self.getActiveMaskAsBase64()
        if mask is None:
            print("ERROR: StableGimpfusionPlugin.inpainting")
            raise Exception("Inpainting must use either a selection or layer mask")

        data = {
            "prompt": prompt + " " + settings["prompt"],
            "negative_prompt": negative_prompt + " " +  settings["negative_prompt"],
            "denoising_strength": float(settings["denoising_strength"]),
            "steps": int(settings["steps"]),
            "cfg_scale": float(settings["cfg_scale"]),
            "width": int(width),
            "height": int(height),
            "init_images": [self.getActiveLayerAsBase64()],
            "seed": int(settings["seed"]),
            "mask": mask,
            "resize_mode": resize_mode,
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 10,
            "inpainting_mask_invert": 0,
            "image_cfg_scale": 5,
            "sampler_index": settings["sampler_name"],
        }

        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
            if cn_enabled:
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/img2img", data)
                # last image is the controlnet mask, ignore it
                if len(response["images"]) > 1:
                    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/img2img", data)

            Layers(image, response["images"]).insertTo(image).scale(1.0/rect.scale)
        except Exception as ex:
            print("ERROR: StableGimpfusionPlugin.inpainting")
            raise ex
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

    def textToImage(self, prompt, negative_prompt, cn_enabled, cn_layer):
        image = self.image
        settings = self.settings
        x, y, width, height = self.getSelectionBounds()
        rect = Rect(width, height).fitBetween((384, 384), (1024, 1024))
        width = rect.width
        height = rect.height

        data = {
            "prompt": prompt + " " + settings["prompt"],
            "negative_prompt": negative_prompt + " " +  settings["negative_prompt"],
            "denoising_strength": float(settings["denoising_strength"]),
            "steps": int(settings["steps"]),
            "cfg_scale": float(settings["cfg_scale"]),
            "width": int(width),
            "height": int(height),
            "sampler_index": settings["sampler_name"],
            "seed": int(settings["seed"]),
        }

        try:
            gimp.pdb.gimp_progress_init("", None)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
            if cn_enabled:
                print("ControlNet is enabled")
                data.update({"controlnet_units": [self.getControlNetParams(cn_layer)]})
                response = self.api.post("/controlnet/txt2img", data)
                # last image is the controlnet mask, ignore it
                if len(response["images"]) > 1:
                    response["images"] = response["images"][:-1]
            else:
                response = self.api.post("/sdapi/v1/txt2img", data)

            #disable=pdb.gimp_image_undo_disable(image)
            Layers(image, response["images"]).insertTo(image).scale(1.0/rect.scale).translate((x, y)).addSelectionAsMask()
            #enable = pdb.gimp_image_undo_enable(image)
        except Exception as ex:
            print("ERROR: StableGimpfusionPlugin.textToImage")
            raise ex
        finally:
            gimp.pdb.gimp_progress_end()
            self.cleanup()

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
        cnlayer = ControlNetLayerData(self.image.active_layer)
        cnlayer.save(settings)

    def config(self, mask_blur, denoising_strength, cfg_scale, sampler_index, steps, seed, prompt, negative_prompt, url):
        model = sd_data.get("models")[model]
        settings = {
            "mask_blur": mask_blur,
            "denoising_strength": denoising_strength,
            "cfg_scale": cfg_scale,
            "sampler_name": SAMPLERS[sampler_index],
            "steps": steps,
            "seed": int(seed),
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


class ControlNetLayerData():
    def __init__(self, layer):
        self.name = "ControlNet"
        self.layer = layer
        self.image = layer.image
        self.load()

    def load(self):
        parasite = self.layer.parasite_find(self.name)
        if not parasite:
            self.data = CONTROLNET_DEFAULT_SETTINGS
            pdb.gimp_layer_set_name(self.layer, self.name + " Layer")
        else:
            self.data = json.loads(parasite.data)
        return self.data

    def save(self, data):
        parasite = gimp.Parasite(self.name, gimpenums.PARASITE_UNDOABLE, json.dumps(data))
        self.layer.parasite_attach(parasite)


class Layers():
    def __init__(self, img, images):
        self.image = img
        color = gimp.pdb.gimp_context_get_foreground()
        gimp.pdb.gimp_context_set_foreground((0, 0, 0))

        layers = []
        for image in images:
            filepath = TempFiles().get("generated.png")
            imageFile = open(filepath, "wb+")
            imageFile.write(base64.b64decode(image))
            imageFile.close()
            layer = gimp.pdb.gimp_file_load_layer(img, filepath)
            pdb.gimp_layer_set_name(layer, "Generated Layer")
            layers.append(layer)

        gimp.pdb.gimp_context_set_foreground(color)
        self.layers = layers

    def scale(self, new_scale=1.0):
        if new_scale != 1.0:
            for layer in self.layers:
                gimp.pdb.gimp_layer_scale(layer, int(layer.width * new_scale), int(layer.height * new_scale), True)
        return self

    def translate(self, offset=None):
        if offset is not None:
            for layer in self.layers:
                gimp.pdb.gimp_layer_set_offsets(layer, offset[0], offset[1])
        return self

    def insertTo(self, image=None):
        image = image or self.image
        for layer in self.layers:
            gimp.pdb.gimp_image_insert_layer(image, layer, None, -1)
        return self

    def addSelectionAsMask(self):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(self.image)
        if not non_empty:
            return
        for layer in self.layers:
            mask = layer.create_mask(ADD_SELECTION_MASK)
            layer.add_mask(mask)
        return self


def handleConfig(image, drawable, *args):
    StableGimpfusionPlugin(image).config(*args)

def handleChangeModel(image, drawable, *args):
    StableGimpfusionPlugin(image).changeModel(*args)

def handleImageToImage(image, drawable, *args):
    StableGimpfusionPlugin(image).imageToImage(*args)

def handleImageToImageFromLayersContext(image, drawable, run_type, *args):
    StableGimpfusionPlugin(image).imageToImage(*args)

def handleInpainting(image, drawable, *args):
    StableGimpfusionPlugin(image).inpainting(*args)

def handleInpaintingFromLayersContext(image, drawable, run_type, *args):
    StableGimpfusionPlugin(image).inpainting(*args)

def handleTextToImage(image, drawable, *args):
    StableGimpfusionPlugin(image).textToImage(*args)

def handleTextToImageFromLayersContext(image, drawable, run_type, *args):
    StableGimpfusionPlugin(image).textToImage(*args)

def handleControlNetLayerConfig(image, drawable, *args):
    StableGimpfusionPlugin(image).saveControlLayer(*args)

def handleControlNetLayerConfigFromLayersContext(image, drawable, run_type, *args):
    StableGimpfusionPlugin(image).saveControlLayer(*args)

models = sd_data.get("models", [])
sd_model_checkpoint = sd_data.get("sd_model_checkpoint")

PLUGIN_FIELDS_IMAGE = [
        (PF_IMAGE, "image", "Image", None),
        (PF_DRAWABLE, "drawable", "Drawable", None),
        ]

PLUGIN_FIELDS_LAYERS = [
        (PF_IMAGE, "image", "Image", None),
        (PF_LAYER, "layer", "Layer", None),
        (PF_INT32, "run_type", "Ignore", 0),
        ]

PLUGIN_FIELDS_COMMON = [
        (PF_TEXT, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
        (PF_TEXT, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"])
        ]

PLUGIN_FIELDS_CONTROLNET = [
        (PF_TOGGLE, "cnet_enabled", "Enable ControlNet", False),
        (PF_LAYER, "cnet_layer", "ControlNet Layer", None)
        ]

PLUGIN_FIELDS_CONFIG = [
    (PF_INT8, "mask_blur", "Mask Blur", 4),
    (PF_SLIDER, "denoising_strength", "Denoising Strength", STABLE_GIMPFUSION_DEFAULT_SETTINGS["denoising_strength"], (0.0, 1.0, 0.1)),
    (PF_SLIDER, "cfg_scale", "CFG Scale", STABLE_GIMPFUSION_DEFAULT_SETTINGS["cfg_scale"], (0, 20, 0.5)),
    (PF_OPTION, "sampler_index", "Sampler", SAMPLERS.index(STABLE_GIMPFUSION_DEFAULT_SETTINGS["sampler_name"]), SAMPLERS),
    (PF_SLIDER, "steps", "Steps", STABLE_GIMPFUSION_DEFAULT_SETTINGS["steps"], (10, 150, 1)),
    (PF_STRING, "seed", "Seed (optional)", STABLE_GIMPFUSION_DEFAULT_SETTINGS["seed"]),
    (PF_STRING, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
    (PF_STRING, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"]),
    (PF_STRING, "api_base", "Backend API URL base", STABLE_GIMPFUSION_DEFAULT_SETTINGS["api_base"]),
    ]

PLUGIN_FIELDS_CHECKPOINT = [
    (PF_OPTION, "model", "Model", models.index(sd_model_checkpoint), models)
    ]

PLUGIN_FIELDS_TXT2IMG = [] + PLUGIN_FIELDS_COMMON + PLUGIN_FIELDS_CONTROLNET

PLUGIN_FIELDS_RESIZE_MODE = [(PF_OPTION, "resize_mode", "Resize Mode", 0, tuple(RESIZE_MODES.keys()))]
PLUGIN_FIELDS_IMG2IMG = [] + PLUGIN_FIELDS_RESIZE_MODE + PLUGIN_FIELDS_COMMON + PLUGIN_FIELDS_CONTROLNET

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
        "<Image>/GimpFusion/Configure",
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
        "<Image>/GimpFusion/Change Model",
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

main()

