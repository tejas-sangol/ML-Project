This dataset contains 1000 images taken from 200 videos of people signing. 5 frames have been extracted from each video. There are 40 videos for each of the digits 1 - 5. These videos have been obtained from 20 different people, over 2 sessions.

You will find 2 folders - raw and cropped.
The file name convention is as follows

<user_id>_<symbol_id>_<session_id_cam1_<frame_id>_<extension>

user_id varies from 1 - 20 as there are 20 users
symbol_id varies from 31 - 35. 31 represents symbol 1, 32 for 2 and so on
session_id can be either 1 or 2.
frame_id varies from 0-4.

Extension will depend on whether the images are cropped or not.
In the raw folder, you will find images organized according to symbols.
These images are of the entire video frame.

The bounding_boxes.csv file tells you where in the frame the hand is present. It contains the top left and bottom right coordinates of the bounding box (guranteed to be a square, but can be of variable size).

For your convenience, these boxes have been cropped out and scaled to size 128x128 and stored in the cropped folder, but with a different extension.

IMPORTANT

For cross validation please do not randomly jumble up the frames. Follow the given procedure.

Assume you are doing 5 fold CV.
Split the user_id list into 5 groups of 4 users each. Now for each group of users, form the training and test sets. So in each fold of the CV, you will train on the data from 16 users and test on the data of the other 4 remaining users. This will give you a more reliable estimate of your classifer's performance.