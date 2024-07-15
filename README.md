<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Graph Creator</h3>

  <p align="center">
    A deep learning Knowlede graph creation pipeline that leverages llm's to create a knowledge graph a at a human level
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Knowledge graphs are a quickly develping powerful medium of representing data. There has been a recent desire to utalize them in large language models. 
The only issue is that Knowledge Graphs take a long time to make and are difficult to update. Using this pipeline a user can create knowledge graphs entirely with unstructed data. EX: websites, pdf's, textfiles. This project also containes one of the only reward functions for automatically generated knowlede graphs in its benchmark section.


<p align="right">(<a href="#readme-top">back to top</a>)</p>






<!-- GETTING STARTED -->
## Getting Started
### Installation

1. Clone the repo
   ```sh
   pip install git+https://github.com/ChristopherOosthuizen/GraphCreator.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
### Making a new Graph
```
import GraphCreation.GraphCreation as gc
import os
os.environ['OPENAI_API_KEY']= <API KEY>
gc.LLM.set_model("gpt-4o")
chunks,g = gc.create_KG_from_url("https://en.wikipedia.org/wiki/Knight_of_the_shire",compression=0.8)
print(gc.LLM.graphquestions(g,"what where the knights of the shire?"))
```
### To Use Exiting Graphs
```
import networkx as nx
g = nx.read_graphml("path to graph")
print(GraphCreation.LLM.graphquestions(g,"Who is Mckenna Grace"))
```
### Make Graph from pdf
```
chunks,g = gc.create_KG_from_pdf("file.pdf",output_file="file to output")
print(GraphCreation.LLM.graphquestions(g,"question about graph"))
```
### Make graph from folder
```
chunks,g = gc.create_KG_from_pdf("file.pdf",output_file="file to output")
print(GraphCreation.LLM.graphquestions(g,"question about graph"))
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Christopher Oosthuizen  - christopher.jan.oosthuizen@gmail.com

Project Link: [https://github.com/ChristopherOosthuizen/GraphCreator](https://github.com/ChristopherOosthuizen/GraphCreator)

<p align="right">(<a href="#readme-top">back to top</a>)</p>