<a id="readme-top"></a>

<!-- Github icons -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<br/>

<!-- Title page -->
<div align="center">
  <h2 align="center">HeraMod</h2>

  <p align="center">
    LLM designed to help, motivate (or demotivate) people (take responses with a grain of salt).
    <br/>
    <br/>
    <a href="https://huggingface.co/spaces/BartoszB/HeraMod">View Demo</a>
    ·
    <a href="https://github.com/BudzioT/HeraMod/issues/new?labels=Bug">Report Bug</a>
    ·
    <a href="https://github.com/BudzioT/HeraMod/issues/new?labels=Feature">Request Feature</a>
  </p>
</div>

<!-- Table of contents -->
<details>
  <summary>Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- About the project -->
## About The Project

![image](https://github.com/user-attachments/assets/ccca6bfd-3092-4279-803e-3fdb41979dde)
<br>
![image](https://github.com/user-attachments/assets/39a99715-0ce8-4432-b0c6-c3450b5e6e27)
<br>
![image](https://github.com/user-attachments/assets/061978d3-e6a5-46b0-9237-4bc461b7540e)

HeraMod is an LLM that was trained on couple of self-motivational, self-improvement related texts. It is supposed to generate kind, motivating words, that might help everyone.

HeraMod is short of Hera Model, Hera is goddess of:
* Childbirth - which is considered something awesome, giving new life is a hard, yet rewarding task - this LLM generates amazing words based on hard prompts
* Family - we feel safe around family, can do anything thanks to them. HeraMod is supposed to give you the same feeling - you are limited only by your own mind!
* Marriage - it is something that influences our entire lifes. Same goes for this model - a small answer might change someone's experience with life entirely.

It isn't yet written in the best way, it's made for scratch - no pretrained weights. Thus responses might not be sometimes adequate to the topic. It will be trained further and improved.

<p align="right">(<a href="#readme-top">Top</a>)</p>


### Built With

* [![Python][Python-logo]][Python-url]
* [![PyTorch][Pytorch-logo]][Pytorch-url]

<p align="right">(<a href="#readme-top">Top</a>)</p>


<!-- Getting started -->
## Getting Started

Here is an example on how to set up your project locally

### Prerequisites

Install all requirements from requirements.txt

### Running

Run your program in any Python IDE or editor of your liking use:
* App.py - for Gradio interface
* gptGenerate - to generate a prompt from terminal
* gptTrain - to train Model on text files from resources/data directory

<p align="right">(<a href="#readme-top">Top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Here are example prompts (LLM will finish them):

<p align="right">(<a href="#readme-top">Top</a>)</p>



[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/BudzioT/HeraMod/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/BudzioT/HeraMod/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/BudzioT/HeraMod/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/bartosz-budnik-19b4b9292
[Python-logo]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[PyTorch-logo]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
