[![GitHub license](https://img.shields.io/github/license/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion) [![GitHub stars](https://img.shields.io/github/stars/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion)  [![awesomeness](https://img.shields.io/badge/awesomeness-maximum-red.svg)](https://github.com/ArtBIT/stable-gimpfusion)

# Stable-Gimpfusion 

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/icon.png" width="200" />


This is a simple Gimp plugin that allows you to augment your painting using [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)) from your local StableDiffusion server.

# Requirements
 - [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running)

# Installation
 - Download the script file [stable-gimpfusion.py](https://github.com/artbit/stable-gimpfusion/raw/master/stable-gimpfusion.py), 
 save into your gimp plugin directory, ie: `~/.gimp-2.10/plug-ins/stable-gimpfusion.py`
 - Restart Gimp, and you will see a new AI menu item
 - Run script via `AI -> Stable Gimpfusion -> Config` and set the backend API URL base (should be `http://127.0.0.1:7860/sdapi/v1/` by default)

# License

[MIT](LICENSE.md)
