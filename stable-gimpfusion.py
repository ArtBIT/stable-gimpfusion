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
        "denoising_strength": 0.8,
        "cfg_scale": 7.5,
        "steps": 50,
        "seed": -1,
        "prompt": "",
        "negative_prompt": "",
        "api_base": "http://127.0.0.1:7860/"
        }

RESIZE_MODES = {
        "Just Resize": 0,
        "Crop And Resize": 1,
        "Resize And Fill": 2,
        "Just Resize (Latent Upscale)": 3
        }

CONTROLNET_DEFAULT_SETTINGS = {
      "input_image": None,
      "mask": None,
      "module": "none",
      "model": "None",
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

PLUGIN_FIELDS_COMMON = [
        (PF_TEXT, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
        (PF_TEXT, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"])
        ]

def roundToMultiple(value, multiple):
    return multiple * math.floor(value/multiple)


class ApiClient():
    def __init__(self, base_url):
        self.setBaseUrl(base_url)

    def setBaseUrl(self, base_url):
        self.base_url = base_url

    def post(self, endpoint, data=None, params=None, headers=None):
        gimp.pdb.gimp_progress_init("", None)
        try:
            data = json.dumps(data)
            headers = headers or {"Content-Type": "application/json", "Accept": "application/json"}
            url = self.base_url + endpoint + "?" + urllib.urlencode(params or {})
            request = urllib2.Request(url=url, data=data, headers=headers)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
            response = urllib2.urlopen(request)
            gimp.pdb.gimp_progress_set_text("Processing response...")
            data = response.read()
            data = json.loads(data)
            gimp.pdb.gimp_progress_end()
            return data
        except Exception as ex:
            raise ex
        finally:
            gimp.pdb.gimp_progress_end()

    def get(self, endpoint, params=None, headers=None):
        gimp.pdb.gimp_progress_init("", None)
        try:
            params = json.dumps(params)
            headers = headers or {"Content-Type": "application/json", "Accept": "application/json"}
            url = self.base_url + endpoint + "?" + urllib.urlencode(params or {})
            request = urllib2.Request(url=url, params=params, headers=headers)
            gimp.pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
            response = urllib2.urlopen(request)
            gimp.pdb.gimp_progress_set_text("Processing response...")
            data = response.read()
            data = json.loads(data)
            gimp.pdb.gimp_progress_end()
            return data
        except Exception as ex:
            raise ex
        finally:
            gimp.pdb.gimp_progress_end()


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


class StableGimpfusionPlugin():
    def __init__(self, image):
        self.name = "stable_gimpfusion"
        self.image = image
        self.loadSettings()
        self.api = ApiClient(self.settings["api_base"])
        self.files = TempFiles()

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

    def getActiveLayerAsBase64(self):
        filepath = self.files.get('init.png')
        self.drawableToFile(self.image.active_layer, filepath)
        return self.fileToBase64(filepath)

    def getActiveMaskAsBase64(self):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(self.image)
        layer = self.image.active_layer
        if non_empty:
            layer = gimp.Layer(self.image, "mask", self.image.width, self.image.height, RGBA_IMAGE, 100, NORMAL_MODE)
            mask = layer.create_mask(ADD_SELECTION_MASK)
            layer.add_mask(mask)
            filepath = self.files.get('selection.png')
            self.drawableToFile(layer.mask, filepath)
            return self.fileToBase64(filepath)
        elif layer.mask:
            filepath = self.files.get('mask.png')
            self.drawableToFile(layer.mask, filepath)
            return self.fileToBase64(filepath)
        else:
            return None

    def getSelectionBounds(self):
        non_empty, x1, y1, x2, y2 = gimp.pdb.gimp_selection_bounds(self.image)
        if non_empty:
            return x1, y1, x2-x1, y2-y1
        return 0, 0, self.image.width, self.image.height

    def cleanup(self):
        self.files.removeAll()
        self.checkUpdate()


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


class ControlNetLayer():
    def __init__(self, layer):
        self.name = "ControlNet"
        self.layer = layer
        self.image = layer.image
        self.loadSettings()

    def loadSettings(self):
        parasite = self.layer.parasite_find(self.name)
        if not parasite:
            self.settings = STABLE_GIMPFUSION_DEFAULT_SETTINGS
        else:
            self.settings = json.loads(parasite.data)

    def saveSettings(self, settings):
        parasite = gimp.Parasite(self.name, gimpenums.PARASITE_UNDOABLE, json.dumps(settings))
        self.image.parasite_attach(parasite)


class ApiLayers():
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


def handleConfig(image, drawable, mask_blur, denoising_strength, cfgScale, steps, seed, prompt, negative_prompt, url):
    settings = {
        "mask_blur": mask_blur,
        "denoising_strength": denoising_strength,
        "cfg_scale": cfgScale,
        "steps": steps,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "api_base": url
    }
    plugin = StableGimpfusionPlugin(image)
    plugin.saveSettings(settings)


def handleImageToImage(image, drawable, prompt, negative_prompt):
    plugin = StableGimpfusionPlugin(image)
    settings = plugin.settings
    rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
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
        "init_images": [plugin.getActiveLayerAsBase64()],
        "seed": settings["seed"]
    }
    try:
        response = plugin.api.post("sdapi/v1/img2img", data)
        ApiLayers(image, response["images"]).insertTo(image).scale(1.0/rect.scale)
    except Exception as ex:
        raise ex
    finally:
        plugin.cleanup()


def handleInpainting(image, drawable, resize_mode, prompt, negative_prompt):
    plugin = StableGimpfusionPlugin(image)
    settings = plugin.settings
    rect = Rect((image.width, image.height)).fitBetween((384, 384), (1024, 1024))
    width = rect.width
    height = rect.height

    mask = plugin.getActiveMaskAsBase64()
    if mask is None:
        raise Exception("Inpainting must use either a selection or layer mask")

    data = {
        "prompt": prompt + " " + settings["prompt"],
        "negative_prompt": negative_prompt + " " +  settings["negative_prompt"],
        "denoising_strength": float(settings["denoising_strength"]),
        "steps": int(settings["steps"]),
        "cfg_scale": float(settings["cfg_scale"]),
        "width": int(width),
        "height": int(height),
        "init_images": [plugin.getActiveLayerAsBase64()],
        "seed": settings["seed"],
        "mask": mask,
        "resize_mode": resize_mode,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 10,
        "inpainting_mask_invert": 0,
        "image_cfg_scale": 5,
    }

    try:
        response = plugin.api.post("sdapi/v1/img2img", data)
        ApiLayers(image, response["images"]).insertTo(image).scale(1.0/rect.scale)
    except Exception as ex:
        raise ex
    finally:
        plugin.cleanup()


def handleTextToImage(image, drawable, prompt, negative_prompt):

    plugin = StableGimpfusionPlugin(image)
    settings = plugin.settings
    x, y, width, height = plugin.getSelectionBounds()
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
        "seed": settings["seed"],
    }

    try:
        response = plugin.api.post("sdapi/v1/txt2img", data=data)
        ApiLayers(image, response["images"]).insertTo(image).scale(1.0/rect.scale).translate((x, y)).addSelectionAsMask()
    except Exception as ex:
        raise ex
    finally:
        plugin.cleanup()

class StableDiffusionData():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StableDiffusionData, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.name = "stable_diffusion_data"
        print("Initializing " + self.name)
        try:
            pprint(shelf)
            pprint(shelf.has_key)
            pprint(self.name)
            settings = self.fetch()
            pprint(settings)
            self.settings = json.loads(settings)
        except Exception as ex:
            pprint("error")
            pprint(ex)

    def fetch(self):
        pprint("Fetching ")
        api = ApiClient(self.settings["api_base"])

        settings = [] + STABLE_GIMPFUSION_DATA
        pprint(settings)
        try:
            cn_models = api.get("/controlnet/model_list")
            pprint(cn_models)
        except Exception as ex:
            raise ex

        return settings

    def get(self, name):
        pprint("Get "+name)
        if name in self.settings:
            return self.settings[name]
        return None


data = StableDiffusionData()


register(
        "stable-gimpfusion-config",
        "This is where you configure params that are shared between all API requests",
        "Gimp Client for the StableDiffusion Automatic1111 API",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/AI/Stable GimpFusion/Global Config",
        "*",
        [
            (PF_INT8, "mask_blur", "Mask Blur", 4),
            (PF_SLIDER, "denoising_strength", "Denoising Strength", STABLE_GIMPFUSION_DEFAULT_SETTINGS["denoising_strength"], (0.0, 1.0, 0.1)),
            (PF_SLIDER, "cfg_scale", "CFG Scale", STABLE_GIMPFUSION_DEFAULT_SETTINGS["cfg_scale"], (0, 20, 0.5)),
            (PF_SLIDER, "steps", "Steps", STABLE_GIMPFUSION_DEFAULT_SETTINGS["steps"], (10, 150, 1)),
            (PF_STRING, "seed", "Seed (optional)", STABLE_GIMPFUSION_DEFAULT_SETTINGS["seed"]),
            (PF_STRING, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
            (PF_STRING, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"]),
            (PF_STRING, "api_base", "Backend API URL base", STABLE_GIMPFUSION_DEFAULT_SETTINGS["api_base"])
            ],
        [],
        handleConfig
        )

register(
        "stable-gimpfusion-txt2img",
        "Text to image",
        "Text to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/AI/Stable GimpFusion/Text to image",
        "*",
        []+PLUGIN_FIELDS_COMMON,
        [],
        handleTextToImage
        )


register(
        "stable-gimpfusion-img2img",
        "Image to image",
        "Image to image",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/AI/Stable GimpFusion/Image to image",
        "*",
        []+PLUGIN_FIELDS_COMMON,
        [],
        handleImageToImage
        )

register(
        "stable-gimpfusion-inpainting",
        "Inpainting",
        "Inpainting",
        "ArtBIT",
        "ArtBIT",
        "2023",
        "<Image>/AI/Stable GimpFusion/Inpainting",
        "*",
        [(PF_OPTION, "resize_mode", "Resize Mode", 0, tuple(RESIZE_MODES.keys()))] + PLUGIN_FIELDS_COMMON,
        [],
        handleInpainting
        )

main()

