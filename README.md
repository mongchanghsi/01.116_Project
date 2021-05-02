## OCR for inventory management

#### Google Sheet Set Up

1. Follow instructions from https://www.youtube.com/watch?v=cnPlKLEGR7E to generate your credentials.json
2. Drop the credentials.json into the directory folder
3. Alternatively, DM me for my credentials.json

### Installation requirements
1. Ensure binaries for tesseract are installed:

    For Linux:
    ```
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install libtesseract-dev
    ```

    For Windows: \
    Refer to https://tesseract-ocr.github.io/tessdoc/Downloads 

2. Install python libraries 
    ```
    pip install -r requirements.txt
    ```

3. Ensure opencv is installed

#### How to use

1. open cmd/terminal
2. cd to directory
3. run `python3 main.py -i test_images/<IMAGE NAME INCLUDING EXT>` for only 1 image (for e.g. `python3 main.py -i test_images/test4_top_cropped.png`)
4. run `python3 main.py -i1 test_images/<IMAGE NAME INCLUDING EXT> -i2 test_images/<IMAGE NAME INCLUDING EXT>` for 2 images (for e.g. `python3 main2.py -i1 test_images/test2_back_cropped.png -i2 test_images/test2_top_cropped.png`)

### Evaluate
* Run 
* Run `python3 evaluate.py -d <path-to-dataset-director>` 
    * E.g. `python3 evaluate.py -d  /home/hwlee96/SUTD/01.116/project/Data`
#### Note

In the directory folder, I have 2 sub-folders; test_images and processed_images.
The test_images are the cropped and rotated images that you want to run OCR on.
The processed_images are the images that will be processed by the script and save it in there.

