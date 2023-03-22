[![GitHub license](https://img.shields.io/github/license/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion) [![GitHub stars](https://img.shields.io/github/stars/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion)  [![awesomeness](https://img.shields.io/badge/awesomeness-maximum-red.svg)](https://github.com/ArtBIT/stable-gimpfusion)

# Stable-Gimpfusion 

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/icon.png" width="200" />

This is a simple Gimp plugin that allows you to augment your painting using [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) from your local StableDiffusion server.

# Example usage

[![Stable Gimpfusion Demo](https://img.youtube.com/vi/eJg_-xqu5OY/0.jpg)](https://www.youtube.com/watch?v=eJg_-xqu5OY)


# Requirements
 - [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running)
 - [Mikubill's StableDiffusion Web-UI ControlNet](https://github.com/Mikubill/sd-webui-controlnet)

# Installation
 - Download the script file [stable-gimpfusion.py](https://github.com/artbit/stable-gimpfusion/raw/master/stable-gimpfusion.py), 
 save into your gimp plug-ins directory, ie: 
   - Linux: `$HOME/.gimp-2.10/plug-ins/` or `$XDG_CONFIG_HOME/GIMP/2.10/plug-ins/`
   - Windows: `%APPDATA%\GIMP\2.10\plug-ins\` or `C:\Users\{your_id}\AppData\Roaming\GIMP\2.10\plug-ins\`
   - OSX: `$HOME/Library/GIMP/2.10/plug-ins/` or `$HOME/Library/Application Support/GIMP/2.10/`
 - Restart Gimp, and you will see a new AI menu item
 - Run script via `AI -> Stable Gimpfusion -> Config` and set the backend API URL base (should be `http://127.0.0.1:7860/sdapi/v1/` by default)

# Troubleshooting
 - Make sure you're running the Automatic1111's Web UI in API mode [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) try accessing [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs) to make sure it's running

# License
[MIT](LICENSE.md)
