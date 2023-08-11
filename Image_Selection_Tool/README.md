# Select False positives for a novel classifier
_Evaluate approaches for which we do not have any groundtruth yet._
1. create a true positive (TP) set from highlight collections. make sure to hit as many positives as possible.
- run the classifier, it should fire almost all the time. the remainder are FN false negatives (FN).
2. run the classifier on unannotated data (e.g. CG data)
- here we are not sure how many positives/negatives there should even be
- use this tool to annotate all FP false positives. everything else is true negatives (TN)

## Usage
```
python server.py --folder <folder/of/local/images/to/annotate>
```

launches a webpage that allows you to toggle annotate images quickly (see below). results are stored in `selected_images.json`
- highlights persist between runs of the server.
- pages with any annotations are highlighted
- 200 images per page

![image](https://github.com/combatiq/false_positive_select/assets/1063330/92d80b16-70e7-4b4e-b277-a0c23d3f4771)


## Tools
to quickly create images, use 
```
./split_videos.sh <input-video> <output-folder>"
```
to split a video into

## ToDo
- [ ] allow for multiple different folders to be annotated.
- [ ] extend to videos
