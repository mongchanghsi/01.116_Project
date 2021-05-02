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
* Script to evaluate Stage 1 - 3 of proposed solution through word-level and character-level accuracy 
* Run `python3 evaluate.py -p <preprocess-type> -d <path-to-dataset-director>` E.g.
    * `python3 evaluate.py -p all -d "/path/to/root/directory"`
    * -d represents dataset directory which is to be in the structure of:
        ```
        root/
            Sample_Images_Data_Dictionary_05032021.csv
            Sensar/
                back/
                    img.jpg
                side/
                    img.jpg
            Tecnis/
                back/
                    img.jpg
                side/
                    img.jpg
            AcrySof/
                back/
                    img.jpg
                side/
                    img.jpg
        ```
    * -p refers to the usage of preprocessing filter, for the best results, we use (Gaussian Blur + Threshold) and Median Blur, hence we recommend to enter the value `all` for this flag.
* Final Word Accuracy will be presented on the terminal: E.g.
    ```
    Stage 1: Total Word Accuracy - 0.3321554770318021 - Total Character Accuracy - 0.31877627276973375
    Stage 2: Total Word Accuracy - 0.49823321554770317 - Total Character Accuracy - 0.5469406819243344
    Stage 3: Total Word Accuracy - 0.4752650176678445 - Total Character Accuracy - 0.5611863615133116
    ```

#### Note

In the directory folder, I have 2 sub-folders; test_images and processed_images.
The test_images are the cropped and rotated images that you want to run OCR on.
The processed_images are the images that will be processed by the script and save it in there.

