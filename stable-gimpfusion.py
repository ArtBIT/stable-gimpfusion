# vim: set noai ts=4 sw=4 expandtab
#!/usr/bin/python

# Stabel Gimpfusion 
# v1.0.0
# Thin API client for Automatic1111's StableDiffusion API
# https://github.com/AUTOMATIC1111/stable-diffusion-webui

import base64
import gimp
import gimpenums
import json
import math
import os
import random
import re
import tempfile
import urllib2
from gimpfu import *
from pprint import pprint

VERSION = 1
PLUGIN_NAME = "stable_gimpfusion"
PLUGIN_VERSION_URL = "https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/main/version.json"

RESIZE_MODES = {
        "Just Resize": 0,
        "Crop And Resize": 1,
        "Resize And Fill": 2,
        "Just Resize (Latent Upscale)": 3
        }

TEMP_FILENAMES = ['mask', 'init', 'generated']
TEMP_FILES = dict(zip(TEMP_FILENAMES, map(lambda f: r"{}".format(os.path.join(tempfile.gettempdir(), f + ".png")), TEMP_FILENAMES)))

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
        "Waiting for the digital paint to dry...",
        ];

STABLE_GIMPFUSION_DEFAULT_SETTINGS = {
        "denoising_strength": 0.8,
        "cfg_scale": 7.5,
        "steps": 50,
        "seed": "",
        "prompt": "",
        "negative_prompt": "",
        "api_base": "http://127.0.0.1:7860/sdapi/v1/"
        }

def scaleRectTo(source, target, maximum=True):
    source_width = source[0]
    source_height = source[1]
    target_width = target[0]
    target_height = target[1]
    aspect_ratio = float(source_width) / source_height
    new_width = makeMultipleOf(64, int(target_height * aspect_ratio))
    new_height = makeMultipleOf(64, int(target_width / aspect_ratio))
    new_scale = float(new_width) / source_width
    if maximum or (new_width >= target_width):
        return (new_width or 1, target_height), new_scale
    return (target_width, new_height or 1), new_scale


def containRect(source, target):
    return scaleRectTo(source, target, True)


def coverRect(source, target):
    return scaleRectTo(source, target, False)


def fitRectBetween(rect, min_rect, max_rect):
    if rect[0] > max_rect[0] or rect[1] > max_rect[1]:
        result, scale = containRect(rect, max_rect)
        return result
    elif rect[0] < min_rect[0] or rect[1] < min_rect[1]:
        result, scale = coverRect(rect, min_rect)
    else:
        result = rect
        scale = 1.0
    return result, scale


def makeMultipleOf(mul, value):
    if value % mul != 0:
        return math.floor(value/mul) * mul
    return value


def checkUpdate():
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
                pdb.gimp_message(data["message"])
        except Exception as ex:
            ex = ex


def cleanup():
    try:
        for tmpfile in TEMP_FILES:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
    except Exception as ex:
        ex = ex


def activeLayerToFile(image, file):
    pdb.file_png_save(image, image.active_layer, file, file, False, 9, True, True, True, True, True)


def activeLayerMaskToFile(image, file):
    layer = image.active_layer
    pdb.file_png_save(image, layer.mask, file, file, False, 9, True, True, True, True, True)


def getSelectionRect(image):
    non_empty, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    return non_empty, x1, y1, x2-x1, y2-y1

def activeSelectionToMask(image, layers):
    non_empty, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    if not non_empty:
        return
    for layer in layers:
        selection = pdb.gimp_image_get_selection(image)
        mask=layer.create_mask(ADD_SELECTION_MASK)
        layer.add_mask(mask)


def activeSelectionToFile(image, file):
    layer = gimp.Layer(image, "mask", image.width, image.height, RGBA_IMAGE, 100, NORMAL_MODE)
    selection = pdb.gimp_image_get_selection(image)
    mask=layer.create_mask(ADD_SELECTION_MASK)
    layer.add_mask(mask)
    pdb.file_png_save(image, layer.mask, file, file, False, 9, True, True, True, True, True)


def fileToBase64(filepath):
    file = open(filepath, "rb")
    return base64.b64encode(file.read())


def base64imageToLayer(img, image):
    imageFile = open(TEMP_FILES["generated"], "wb+")
    imageFile.write(base64.b64decode(image))
    imageFile.close()
    return pdb.gimp_file_load_layer(img, TEMP_FILES["generated"])


def convertApiImagesToLayers(img, images):
    color = pdb.gimp_context_get_foreground()
    pdb.gimp_context_set_foreground((0, 0, 0))

    layers = []
    for image in images:
        layers.append(base64imageToLayer(img, image))
    pdb.gimp_context_set_foreground(color)
    return layers


def scaleLayers(img, layers, scale = 1.0):
    if scale != 1.0:
        for layer in layers:
            pdb.gimp_layer_scale(layer, int(layer.width * scale), int(layer.height * scale), True)


def offsetLayers(img, layers, offset = None):
    if offset is not None:
        for layer in layers:
            pdb.gimp_layer_set_offsets(layer, offset[0], offset[1])


def insertLayers(img, layers):
    for layer in layers:
        pdb.gimp_image_insert_layer(img, layer, None, -1)


def loadImageSettings(image):
    parasite = image.parasite_find(PLUGIN_NAME)
    if not parasite:
        return STABLE_GIMPFUSION_DEFAULT_SETTINGS
    return json.loads(parasite.data)


def saveImageSettings(image, settings):
    parasite = gimp.Parasite(PLUGIN_NAME, gimpenums.PARASITE_UNDOABLE, json.dumps(settings))
    image.parasite_attach(parasite)

def apiRequest(url, data):
    data = json.dumps(data)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request = urllib2.Request(url=url, data=data, headers=headers)

    pdb.gimp_progress_init("", None)
    try:
        pdb.gimp_progress_set_text(random.choice(GENERATION_MESSAGES))
        response = urllib2.urlopen(request)
        pdb.gimp_progress_set_text("Processing response...")
        data = response.read()
        data = json.loads(data)
        pdb.gimp_progress_end()
        return data
    except Exception as ex:
        raise ex
    finally:
        pdb.gimp_progress_end()

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
    saveImageSettings(image, settings)

def handleImageToImage(image, drawable, prompt, negative_prompt):
    settings = loadImageSettings(image)
    width = image.width
    height = image.height
    min_rect = (384, 384)
    max_rect = (1024, 1024)
    rect, scale = fitRectBetween((width, height), min_rect, max_rect)
    width, height = rect

    if prompt == "":
        raise Exception("Please enter a prompt.")

    if not settings["seed"]:
        seed = random.randint(0, 2**31)
    else:
        seed = int(settings["seed"])

    activeLayerToFile(image, TEMP_FILES["init"])
    init_images = [fileToBase64(TEMP_FILES["init"])]

    params = {
        "prompt": prompt + settings["prompt"],
        "negative_prompt": negative_prompt + settings["negative_prompt"],
        "denoising_strength": float(settings["denoising_strength"]),
        "steps": int(settings["steps"]),
        "cfg_scale": float(settings["cfg_scale"]),
        "width": int(width),
        "height": int(height),
        "init_images": init_images,
        "seed": seed,
    }

    try:
        data = apiRequest(settings["api_base"] + "img2img", params) 
        layers = convertApiImagesToLayers(image, data["images"])
        insertLayers(image, layers)
        scaleLayers(image, layers, 1.0/scale)
    except Exception as ex:
        raise ex
    finally:
        cleanup()
        checkUpdate()

def handleInpainting(image, drawable, resize_mode, prompt, negative_prompt):
    settings = loadImageSettings(image)
    width = image.width
    height = image.height
    min_rect = (384, 384)
    max_rect = (1024, 1024)
    rect, scale = fitRectBetween((width, height), min_rect, max_rect)
    width, height = rect

    if prompt == "":
        raise Exception("Please enter a prompt.")

    if not settings["seed"]:
        seed = random.randint(0, 2**31)
    else:
        seed = int(settings["seed"])

    activeLayerToFile(image, TEMP_FILES["init"])
    init_images = [fileToBase64(TEMP_FILES["init"])]
    non_empty, x1, y1, x2, y2 = pdb.gimp_selection_bounds(image)
    if non_empty:
        activeSelectionToFile(image, TEMP_FILES["mask"])
    elif image.mask:
        activeLayerMaskToFile(image, TEMP_FILES["mask"])
    else:
        raise Exception("Inpainting must use either a selection or layer mask")

    params = {
        "prompt": prompt + settings["prompt"],
        "negative_prompt": negative_prompt + settings["negative_prompt"],
        "denoising_strength": float(settings["denoising_strength"]),
        "steps": int(settings["steps"]),
        "cfg_scale": float(settings["cfg_scale"]),
        "width": int(width),
        "height": int(height),
        "init_images": init_images,
        "seed": seed,
        "mask": fileToBase64(TEMP_FILES["mask"]),
        "resize_mode": resize_mode,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 10,
        "inpainting_mask_invert": 0,
        "image_cfg_scale": 5,
    }

    try:
        data = apiRequest(settings["api_base"] + "img2img", params) 
        layers = convertApiImagesToLayers(image, data["images"])
        insertLayers(image, layers)
        scaleLayers(image, layers, 1.0/scale)
    except Exception as ex:
        raise ex
    finally:
        cleanup()
        checkUpdate()

def handleTextToImage(image, drawable, prompt, negative_prompt):
    settings = loadImageSettings(image)
    non_empty, x, y, width, height = getSelectionRect(image)

    min_rect = (384, 384)
    max_rect = (1024, 1024)
    rect, scale = fitRectBetween((width, height), min_rect, max_rect)
    width, height = rect

    if prompt == "":
        raise Exception("Please enter a prompt.")

    if not settings["seed"]:
        seed = random.randint(0, 2**31)
    else:
        seed = int(settings["seed"])

    params = {
        "prompt": prompt + settings["prompt"],
        "negative_prompt": negative_prompt + settings["negative_prompt"],
        "denoising_strength": float(settings["denoising_strength"]),
        "steps": int(settings["steps"]),
        "cfg_scale": float(settings["cfg_scale"]),
        "width": int(width),
        "height": int(height),
        "seed": seed
    }

    try:
        data = apiRequest(settings["api_base"] + "txt2img", params) 
        layers = convertApiImagesToLayers(image, data["images"])
        insertLayers(image, layers)
        scaleLayers(image, layers, 1.0/scale)
        offsetLayers(image, layers, (x, y))
        activeSelectionToMask(image, layers)
        pdb.gimp_selection_none(image)
    except Exception as ex:
        raise ex
    finally:
        cleanup()
        checkUpdate()

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
        [
            (PF_STRING, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
            (PF_STRING, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"])
            ],
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
        [
            (PF_STRING, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
            (PF_STRING, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"])
            ],
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
        [
            (PF_OPTION, "resize_mode", "Resize Mode", 0, tuple(RESIZE_MODES.keys())),
            (PF_STRING, "prompt", "Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["prompt"]),
            (PF_STRING, "negative_prompt", "Negative Prompt", STABLE_GIMPFUSION_DEFAULT_SETTINGS["negative_prompt"])
            ],
        [],
        handleInpainting
        )

main()

