# About the Project

This project provides a modular architecture to run quantitative analysis of human behaviors starting from visual source.
Thi is a work in progress. Modules to extract audio features will be integrated soon as well. 

This project was developed to provide a compact framework to perform quantitative analysis and features extraction to study humans behavioral dynamics in human-robot-interactions.
It is based on several state-of-the-art pretrained models.


## Getting Started

### Prerequisites

Install all the libraries on the requirements.txt file

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/giulia-belgiovine/behavior-features-extractors.git

## Usage

```console
// Run your application
$ python videoAnalysis.py [OPTIONS] NAME

Options:

  --root       Name of the path where input_videos folder is
  --format     Format of the videos to be processed

  --head       Activate head analysis. It saves in the output file the head direction labels and Euler angles.
  --body       Activate body analysis. It saves in the output file selected joints' coordinates.
  --flow       Activate Optical flow analysis. It saves in the output file the magnitude of motion.
  --emotion    Activate emotion analysis. It saves in the output file the Arousal and Valence values.
  
   --debug      Activate debug modality. It displays and saves output videos with analysis result.
```
For each analysis, it will be possible to use different models. More info about the available models are coming...

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>
