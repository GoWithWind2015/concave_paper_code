# code corresponding to 《Splitting ore from X-ray image based on improved robust concave-point algorithm》
#### creat enviroment using conda 
```shell
conda create --name picture_deal python=3.7
```

#### install pyton development enviroment
```shell
conda activate picture_deal
pip install -r requirements.txt
```

#### run code
```shell
# run the algorithm proposed in this paper
python qie_tu/main.py basic

# simple concave point alogorithm
python qie_tu/main.py simple

# watershed algorithm
python qie_tu/main.py watershed
```

#### test result

result of 2332: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|    24   |         1 |       0|         23 | 
| simple concave point|   24 |          1 |       2|         21| 
|watershed algorithm|    24  |         0 |       0  |       24 | 

result of 2316: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
| algorithm of this paper |  20|           2 |       0  |      18 |
|simple concave point|  20|         1|       2 |       17|
|watershed algorithm|  20 |           3 |        1|          16 |  


result of 2820: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
| algorithm of this paper |  15|           1 |       0  |      14 |
|simple concave point|  15|         1|       0 |       14|
|watershed algorithm|  15 |           2 |        0|          13 |  



result of 5134: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|    1   |         0|         0   |       1 |  
|simple concave point|    1     |       1 |        0 |         14|  
|watershed algorithm  |    1   |         2    |     0    |      13 |  


result of 2307: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper |   29    |        0  |       0     |     29 |  
|simple concave point |   29|            0|         2 |         27|  
|watershed algorithm  |   29 |         0|         0|          29 |  


result of 2815: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|  17    |      0|       0 |       17 |  
|simple concave point  | 17     |     0 |      2   |     15|  
|watershed algorithm    | 17      |    0  |     0  |      17 |  



result of 2357: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper | 26    |      1   |    0   |     25 |  
| simple concave point  | 26    |      1  |     1    |    24 |  
|watershed algorithm  | 26     |     0   |    2     |   24 |  


result of 5819: 
| algorithm |  number of stones | under-segmentatioin|  over-segmentation | segmentation properly |
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|  2         |  0     |  0    |     2 |  
|simple concave point |  2         | 0      |  1    |     1  |  
|watershed algorithm  |  2         | 0      |  0    |     2 |  

Summary:
|algorithm |  number of stones | under-segmentatioin | over-segmentation| segmentation properly  |    
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|  134         |  5     |  0    |     129 |  
|simple concave point |  134         |5      |  10    |     119  |  
|watershed algorithm |  134         | 7      |  3    |     124 |  

Summary(%):

|algorithm |  number of stones | under-segmentatioin(%) | over-segmentation(%)| segmentation properly(%)  |    
| --- |        ---       | ---    |   ---   |  ---     |
|algorithm of this paper|  134         |  3.73     |  0    |     96.27 |  
|simple concave point |  134         | 3.73      |  7.46    |     88.81 |  
|watershed algorithm  |  134         | 5.22      |  2.24    |     92.54 |  


