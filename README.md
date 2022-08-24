# Would planting trees in desert areas reduce the global temperature?

## Problem

Sea level rises, animals go extinct, the frequency of natural disasters increases... Those are only a few examples of the problems we face due to an accumulation of human-induced greenhouse gas emissions. Climate change is a major problem of our times. The global average temperature (GAT) keeps rising and could lead to reaching irreversible tipping points, such as the melting of the Greenland ice sheets [[13]](#13). The Paris Agreement aims to keep GAT below 1.5 degrees but how can we reach this goal?

## Solution

To mitigate climate change one solution is to plant a trillion trees in the Sahara Desert. This is similar to the number of Goymer, who estimated that we would need 1.2 trillion trees to absorb the excess CO2 [[7]](#7). To put this into perspective, currently, there are around 3 trillion trees on earth, this is half the amount that existed before human civilization [[3]](#3).

In this report, we estimate the impact of this mitigation strategy with the help of data and machine learning techniques. This data-driven approach aims to find a way to counteract climate change. We analyse the effect of planting trees on the GAT through three channels. The first one is task A, where we analyse how trees sequestrate CO2 which would then lead to less CO2 in the atmosphere and eventually decrease the temperature. The second channel is the albedo change looked at  in task B. The desert changes from a light brown to dark green, this affects the reflective capabilities of the earth and thus temperature. Further, in task C we will analyse the impact on the water cycle which again will  change temperature. Finally, we put these three channels together to analyze the overall impact of planting trees on GAT and give a policy recommendation. We would like to note that planting trees is not sufficient, we also need to reduce our emissions.

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186128378-0dceb679-de76-486a-b657-23cee8f8e964.png" width="600">
</p>

## Task A: Model and predict the GAT based on updated CO2 emissions if we plant 1 trillion trees.

In this part, we will model and predict the GAT. The aim is to find the impact of panting 1 trillion trees on CO2 sequestration. We examine different scenarios to determine if planting a trillion trees is a viable solution to tackle climate change. We would like to note that we are aware of the controversy of this topic but we aim to understand the situation and predict GAT based on these different scenarios.

The three main scenarios are:
1. Business as usual, meaning a continuous increase of emissions
2. Realistic decrease of emissions to 69%
3. Net-zero by 2050
4. Best rate, decrease emissions to 10%

For each of these scenarios, we model what would happen if:

a. We do not plant any trees and assume that the number of trees worldwide remains constant.

b. We plant a trillion trees and assume that they sequestrate 42 GtC.

c. We plant a trillion trees and assume that they sequestrate 107 GtC.

In the following grid, each scenario and its predicted plots are depicted. The columns represent no trees, trees that sequestrate 42GtC and 107GtC. The rows represent the scenarios, namely business as usual, decrease to 69%, zero-carbon and decrease to 10%.

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186135110-e67170cc-9f64-41f0-a7ff-b42b46336c02.png" width="1000">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186135205-7e272c68-4dca-4229-89d2-a4514c6b708e.png" width="1000">
</p>

```
Figure 1: Graphs with different scenarios
```

Our approach is purely data-driven since we replicated the model of Mesnage and Vlachos to be able to control parameters, like the reduction of CO2 emissions, the sequestration potential of trees, and its speed [[10]](#10). The aim is to better understand the relationship between these parameters but we had to take many assumptions in order to model the relations. First, we assumed a causal relationship such that a decrease in CO2 levels leads to a decline of the GAT. In reality, we know that a correlation exists but the causality is still debated. Further, many other factors are in play when planting trees, like the albedo and water cycle change modeled later on in this notebook. We also assumed that the ocean's capacity to absorb CO2 stays constant.

Our findings are that fossil-fuel energy production needs to decrease to at least 10% and trees substantially help to reach the goals of the Paris agreement. Thus, the main takeaway is that the climate economy should protect existing forests, and reforestation, as well as afforestation, are important to tackle climate change. These efforts should then be combined with energy transformation to have an average temperature increase of less than 1.5 Degree Celsius.

## Task B: Model and predict the GAT if we additionally consider the change in albedo in the Sahara

We would like to answer questions such as:
It will become less reflective, by how much?
How much more solar radiation will be absorbed?
What is the impact on GAT?
etc

There are many other factors involved when planting trees. The albedo adjustment caused by the change of surface induces a temperature variation because when the trees are grown the land surface becomes darker compared to having ice or sand and hence absorbs more light and heat. It is important to note that (de-)forestation plays a different role at different latitudes and it would be interesting to include this in future studies [[4]](#4).

First of all, we want to know the area of the surface needed to plant 1 trillion trees to then compare it to the surface of the Sahara. From the paper of Rotenberg and  Yakir, we find that they can plant 300 trees in one hectare [[12]](#12). Thus, we can find the surface of one tree and then multiply it by a trillion trees.

The Sahara is 3.66 times smaller than the surface needed to plant 1 trillion trees! Therefore, we look at different projects planning to plant a big amount of trees and which areas they mention in their text.

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186137012-2eb43985-8c24-4d5c-b0f4-f3a1c066c313.png" width="400">
</p>

```
Figure 2: Areas with iniatives to plant trees raphs with different scenarios
```

In the World Cloud, we observe that initiatives to plant trees are active in diverse areas. The focus lies especially on India and the United States. In the following part, we continue with the assumption that the trillion trees are planted in deserts like the Sahara. This is crucial because the albedo of the desert is different than in other regions.

To calculate the influence of planting trees on the worldwide albedo, we use the current albedo of the earth, the Sahara, and a forest. This way we can calculate the change of albedo linked to the plantation of trees [[6]](#6). We also calculate the affected percentual surface of the earth. We train a predictor to calculate the temperature increase due to the albedo change based on a dataset [[11]](#11).

We use the following formula :

Albedo_earth_now = Surface_stable_albedo * Albedo_stable + Surface_changing_albedo * Albedo_changing_before

Then we calculate the albedo that stays stable, which is not influenced by the trees planted.  Afterward, we can determine the new albedo including the trillion new trees and we can find the change of albedo.

We implemented a first-order linear time-invariant system to simulate a gradual increase in temperature. Trees absorb CO2 at different rates, hence the type of tree matters. Mangroves, oaks, and chestnuts are good candidates, but one should keep in mind that species need to be planted according to their local climate [[10]](#10). An Oak normally takes 40 years to grow large [[9]](#9) and a Mangrove 15 [[5]](#5).
Thus, for the time constant (TC) we decided to take the average of the two. Additionally, we assume that after four times the TC, the tree is fully grown. This stems from an analogy often used in engineering, the so-called electrical RC circuit (Figure 3). It assumes that after 1 TC (on the picture RC) the result is around 66% and after 4 TC it almost reached 100%. This is in our eyes a realistic approximation that represents an increase of temperature as the albedo reacts quickly to it and seen from above, a tree's surface grows fast in the beginning and then less with time. But we would like to note that it would be necessary to verify this development in further analysis.

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186344623-1c5ae579-cc2b-4562-a338-7f45a2a6627e.png" width="600">
</p>

```
Figure 3: Analogy made with an electrical RC circuit
```

In the following grid, each scenario and its predicted plots are depicted. It is the same as before but now also includes the impact of the albedo change, shown with the new violet curve called "Temperature predictions w/ Albedo".

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186138529-78fd928e-3ae2-48fa-9b00-ef09433e4b1c.png" width="1000">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186138599-a40c8577-89f7-4a6e-a2cc-87d4f704c6d2.png" width="1000">
</p>

```
Figure 4: Graphs with different scenarios + Albedo
```

## Task C:  Model and predict the GAT if you also consider the water cycle changes.

After a lot of literature research, we decided to focus on the impact of clouds on the GAT. Their presence is directly linked to the water cycle. As more trees are planted, there is more evaporation and thus more clouds [[13]](#13). Here we look a the impact of their albedo on our prediction by reusing the function of  Task B.

For low albedo values, the presence of clouds roughly doubles the original albedo [[8]](#8). The deciduous forest mean from before was 0.175, thus with clouds, the final albedo is 2 * 0.175 = 0.35.

In these plots the temperature prediction w/ Albedo refers only to the albedo change with and without clouds over the forest.

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186139294-7a18679a-3bd2-4b42-92cc-45c4d2c3270f.png" width="1000">
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/110820736/186139429-ee871281-1465-4718-95d9-d67158f071a7.png" width="1000">
</p>

```
Figure 5: Graphs with different scenarios + Albedo + Cloud Coverage
```

We observe that the immediate effect of cloud coverage on the Sahara desert has in the beginning a big negative effect and then becomes parallel to the initial trend with time. This is due to our assumptions like the first-order linear time-invariant system to simulate a gradual decrease in temperature.

## Policy Recommendation

Regarding policy recommendations, we have to be careful. First let's be clear: planting a million trees won't solve the climate crisis. The priority has to be to change our habits in consuming and using energy. We need to stop the growth that has been going on for many decades and start considering other narratives. However, we will need to use technology to help us doing so combined with a cultural change on how we consume and treat nature. One technical aspect on how we can reduce our CO2 emissions are carbon dioxide removal techniques, which could definitely help us to mitigate climate change. Similarly, planting trees is a good thing. It can only be positive as it helps cleaning the air and reduces the global average temperatures. But in this study, we found that even with a trillion trees planted, the sufficiency to reach the Paris Agreement strongly depends on the emission scenario in question.

## Further Research

The amount of carbon sequestrated by planting trees is still uncertain because researchers find extremely different results. This is only one example in which machine learning can help to get more exact estimations of the impact of reforestation.

It would also be helpful to analyze where trees can be grown efficiently and try to understand better the link between albedo change and the GAT. As already mentioned in Task B, a lot of assumptions flow into our modelisation. These assumptions should be verified in future studies. Generally, the GAT and the climate system are influenced by many factors and feedback loops. It is an extremely challenging task to model them all as near to reality as possible but at the same time not overcomplicate it. Hence, climate scientists,engineers,  data scientists, etc. all need to work together to combat climate change.

## Video

You can find a video describing the link and the project here: https://www.youtube.com/watch?v=B9EKW5d_tv0


# Collaborators
- [Mia Frey](https://www.linkedin.com/in/mia-frey-28209a208/)
- [Carla Schmid](https://www.linkedin.com/in/carla-schmid/)
- [Oscar Keren](https://www.linkedin.com/in/oscar-keren-662522248/)

# Sources

<a id="1">[1]</a> Bakken, B. E. (2020). Electricity production is on a sustained charge. Petroleum Economist, Outlook 202.

<a id="2">[2]</a> Bastin, J. F., Finegold, Y., Garcia, C., Mollicone, D., Rezende, M., Routh, D., ... & Crowther, T. W. (2019). The global tree restoration potential. Science, 365(6448), 76-79.

<a id="3">[3]</a> Crowther, T. W., Glick, H. B., Covey, K. R., Bettigole, C., Maynard, D. S., Thomas, S. M., ... & Bradford, M. A. (2015). Mapping tree density at a global scale. Nature, 525(7568), 201-205.

<a id="4">[4]</a> Davin, E. L., & de Noblet-Ducoudré, N. (2010). Climatic impact of global-scale deforestation: Radiative versus nonradiative processes. Journal of Climate, 23(1), 97-112.

<a id="5">[5]</a> Deibus, S. (2020). How long does it take mangroves to grow? Online forum available at https://askinglot.com/how-long-does-it-take-mangroves-to-grow, Accessed on 6. January 2022

<a id="6">[6]</a> Goosse, H., Barriat, P. Y., Lefebvre, W., Loutre, M. F., & Zunz, V. (2010). Introduction to climate dynamics and climate modeling. Online textbook.

<a id="7">[7]</a> Goymer, P. (2018). A trillion trees. Nature ecology & evolution, 2(2), 208-209.

<a id="8">[8]</a> Genyuk, J. (2013). Global Warming, Clouds and Albedo: Feedback Loops. Online blog available at https://www.windows2universe.org/earth/climate/warming_clouds_albedo_feedback.html, Accessed on 6. January 2022

<a id="9">[9]</a> Local Tree Estimates. (n/a). How Long Does it Take for a Tree to Grow? Online blog available at https://localtreeestimates.com/how-long-does-it-take-for-a-tree-to-grow/, Accessed on 6. January 2022

<a id="10">[10]</a> Mesnage, C., & Vlachos, M. (2020). Is a trillion trees enough?. arXiv preprint arXiv:2007.00508.

<a id="11">[11]</a> Parker, B. (2016). Expected Temperature Increase Due to Changes in Either the Earth's Albedo or CO2 Emissions. Online document available at https://ccdatacenter.org/documents/AlbedoCO2TempCalcs.pdf, Accessed on 6. January 2022

<a id="12">[12]</a> Rotenberg, E., & Yakir, D. (2011). Distinct patterns of changes in surface energy budget associated with forestation in the semiarid region. Global change biology, 17(4), 1536-1548.

<a id="13">[13]</a> Schmale, J. (2021). Course of science of climate change

<a id="14">[14]</a> Veldman, J. W., Aleman, J. C., Alvarado, S. T., Anderson, T. M., Archibald, S., Bond, W. J., ... & Zaloumis, N. P. (2019). Comment on "The global tree restoration potential". Science, 366(6463).
