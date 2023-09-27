[![GitHub license](https://img.shields.io/github/license/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion) [![GitHub stars](https://img.shields.io/github/stars/ArtBIT/stable-gimpfusion.svg)](https://github.com/ArtBIT/stable-gimpfusion) [![awesomeness](https://img.shields.io/badge/awesomeness-maximum-red.svg)](https://github.com/ArtBIT/stable-gimpfusion)

# Stable-Gimpfusion

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/icon.png" width="200" />

This is a simple Gimp plugin that allows you to augment your painting using [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) from your local StableDiffusion server.

# Example usage

Using Mona Lisa as the ControlNet

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/monalisa.png" width="400" />

Converting it to Lady Gaga

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/monalisa-controlnet-to-ladygaga.png" width="400" />

And inpainting some nerdy glasses

<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/ladygaga-inpainting.png" width="400" />
<img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/ladygaga-inpainting-result.png" width="400" />

See the demo video on YouTube

<a href="https://www.youtube.com/watch?v=4IuIKe1sEFY" title="Stable Gimpfusion Demo"><img src="https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/master/assets/youtube-icon.jpg" width="100" /></a>

# Requirements

- [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running)
- [Mikubill's StableDiffusion Web-UI ControlNet](https://github.com/Mikubill/sd-webui-controlnet)

# Installation

- Download the script file [stable-gimpfusion.py](https://raw.githubusercontent.com/ArtBIT/stable-gimpfusion/main/stable_gimpfusion.py),
  save into your gimp plug-ins directory, ie:
  - Linux: `$HOME/.gimp-2.10/plug-ins/` or `$XDG_CONFIG_HOME/GIMP/2.10/plug-ins/`
  - Windows: `%APPDATA%\GIMP\2.10\plug-ins\` or `C:\Users\{your_id}\AppData\Roaming\GIMP\2.10\plug-ins\`
  - OSX: `$HOME/Library/GIMP/2.10/plug-ins/` or `$HOME/Library/Application Support/GIMP/2.10/plug-ins`
- Ensure the execute bit is set on MacOS and Linux by running `chmod +x /stable_gimpfusion.py`
- Restart Gimp, and you will see a new AI menu item
- Run script via `AI -> Stable Gimpfusion -> Config` and set the backend API URL base (should be `http://127.0.0.1:7860/` by default)

# Troubleshooting

- Make sure you're running the Automatic1111's Web UI in API mode (`--api`) [Automatic1111's StableDiffusion Web-UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) try accessing [http://127.0.0.1:7860/docs](http://127.0.0.1:7860/docs) and verify the `/sdapi/` routes are present to make sure it's running
- Ensure your Gimp installation has python support (You should see `Filters>Python-fu>Console` in the menu)
- Verify the plugin folder that you are using (~/.config/GIMP/2.20/plug-ins) listed in the GIMP's plug-ins folders. (`Edit>Preferences>Folders>Plug-Ins`)

# License

[MIT](LICENSE.md)
