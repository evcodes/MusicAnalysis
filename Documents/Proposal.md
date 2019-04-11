# Data Mining: Project Plan
### Eddy Varela, Dylan Hoffmann, Meghan Garrity, Nihal Pai


## What is the vision? 


To determine what factors make a song a hit based on various features of the song. We will be looking at data from the U.S. Billboard Top 100.

## Why is the problem interesting?	


This problem could help publishing companies/media distributors/independent artists have a 
better idea of what songs could generate commercial success. Spotify has built tools to help 
artists get a better sense of what their audience is interested in so they could tweak their content 
for maximum engagement. 

## What is the purpose of the project? 

Investigate Metadata on top 100 songs over time to have a general idea if a new song would 
make it on the top 100 billboard. After analyzing the various 

## What results do you expect?

Songs with musical features that correspond to those found in pop music (e.g. moderate - high BPM, strong use of vocals, etc.) are more likely to be ‘hits’ 

Comments: 
Nihal did a project using Spotify’s recommendation API, which classifies songs numerically based on quantifiable song attributes (like each song has a score for how ‘happy’ it sounds for example) 
> https://developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/

Dylan also has a twitter sentiment analysis engine that we can use to analyze how an artist’s twitter interactions correlate with the song’s performance.)
Songs that are published by more established record labels will have a much better chance of making it to the top 100.
Songs that are put out in certain times of the year are more likely perform better (e.g. summer songs about partying with a tropical beat will garner more attention as opposed to the same song being published in the winter)


## What data do you plan to use?


Free Music Archive: A Music Analysis Dataset: 
> https://github.com/mdeff/fma
  
Million Song Dataset: 
> https://labrosa.ee.columbia.edu/millionsong/

Lyric Dataset: 
> http://labrosa.ee.columbia.edu/millionsong/musixmatch

>Billboard Top 100

>Spotify Analytics tools

>Song data 

>Tweets


## How do you plan to gather the data? 

## All of the data we plan to use have APIs or other convenient methods of accessing and importing the data?

## Are the data big enough and of suitable quality?

Yes, the data is plentiful and accessible over a significant period of time. The metadata of songs 
is vast and encapsulates lots of different features that are useful for analysis. All songs in the 
data sets have these features 

## What data preparation steps do you plan to take?

We need to first gather the top 100 songs (American most likely) over some period of time in monthly intervals. Then we need to connect these songs to their data filled counter parts and analyze their features, with their position on the billboard as their label.


## What methodology, what models do you plan to use?

We will have to evaluate which methodologies out of linear regression, knn, etc. gives us the best insight into our data 
- Decision Trees
- Knn
- Convolutional Neural Network
	
## How would you visualize the results?

Unfortunately, we cannot comment on how we plan to visualize our results until we have an
idea of what our results are.