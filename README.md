## OCR for inventory management

#### How to use

1. open cmd/terminal
2. cd to directory
3. run `python3 main.py -i test_images/<IMAGE NAME INCLUDING EXT>` for only 1 image (for e.g. `python3 main.py -i test_images/test4_top_cropped.png`)
4. run `python3 main.py -i1 test_images/<IMAGE NAME INCLUDING EXT> -i2 test_images/<IMAGE NAME INCLUDING EXT>` for 2 images (for e.g. `python3 main2.py -i1 test_images/test2_back_cropped.png -i2 test_images/test2_top_cropped.png`)

#### Note

In the directory folder, I have 2 sub-folders; test_images and processed_images.
The test_images are the cropped and rotated images that you want to run OCR on.
The processed_images are the images that will be processed by the script and save it in there.
