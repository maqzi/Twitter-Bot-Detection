# Twitter-Bot-Detection
## Terminator 6: Humans vs. Bots/Cyborgs (on Twitter)

Identifying bots on twitter using various Machine Learning algorithms. 

## Dataset
Finalized. Uploaded in Repo.

### Wordcloud of User Descriptions
![Descriptions Visualized](https://github.com/maqzi/Twitter-Bot-Detection/blob/master/wordcloud_white.png?raw=true)

### Process
descriptions -> preprocessed+normalized -> naivebayes guesswork 
descriptions -> preprocessed(implement normalization) -> svm guesswork 
naivebayes guesswork + svm guesswork + important numerical features -> [LogR / RF / DT / MNB / BNB] (CrossValidated+GridSearchedForAnOptimizedModel) -> prediction
