

## Synopsis

This folder contains code to simulate a group of objects on top of a table. It loads CAD models, scales accordingly to the size of the table and simulates a scene. Then, it adds textures to all the objects, table and background, before saving images of the scene. Additionally, we provide the code to obtain views from a CAD model and save its rotation with respect to the cameras.

## Use

The codes are prepared to run on Maya 2016, but should run on different versions. The code only depends on the Mental Ray extension, to render high quality images. The scripts can be opened and runned through the Maya script editor. Before starting to render, the script Functions.mel should be run to declare the basic functions.

## Scripts 

### Functions.mel
This script contains the basic functions to render scenes or views, like creating cameras in the general distribution or removing them. This also creates the table plane and the RGB and depth render layers. However, after running this program the render layer has to be set to render depth. This cannot be automatized in Maya and the instructions are indicated in the code.

### SaveSceneImages.mel

Script to simulate scenes and obtain images. By default, only simulates one scene. The paths to the CAD models and to the folder where the data is saved should be changed.


### Obtain_80views.mel

Code to obtain the first discretization level, which is based on a dodecahedron structure. This sets 20 points in the 3D sphere, in which four cameras with different roll angle are created. The total number of cameras is 80.

### Obtain_168views.mel

In this case, we obtain a finer discretization by placing an icosahedron and dividing its faces. Since the icosahedron faces are triangles, they can be divided continuously to get more points. With one division, we set an extra vertex per edge, for a total of 42 vertex. With every 4 roll angles in each vertex, the total number of views is 168.

### Obtain_648views.mel

Similar to the previous code, with another division of the triangular faces. 

### SaveViews.mel

After placing one of the three previous discretizations, this script obtains views of one or more CAD models. The path to the CAD models has to be changed.

