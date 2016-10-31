Improving travel time predictions with bike share data
======================================================

While directions services like Google Maps are very good at picking the best route and estimating travel time for driving and public transit directions, at least in my experience, I have found that this is not the case for bicycling directions.
There are many reasons for this, one of which is that it is much harder to get good data from users than it is for other modes.
Data from bike share services such as Citibike offer a much larger amount of data than might have been previously available.

I hope to use data from Citibike and other bike share systems, as well as data from Google Maps to predict travel times as measured by the bike share system.
These can be compared to predictions made by Google Maps or other directions services.
I may also include other data sources, such  as weather data, which has been shown by multiple sources to be a good predictor of ridership.

As a proof of concept, I looked at a subset of data with which I am personally familiar.
Along the west side of Manhattan, adjacent to the Hudson River, is a bicycle and pedestrian path called the Hudson River Greenway.
Since this path is separated from car traffic and has nearly zero traffic lights, I would imagine that traffic patterns and travel speeds would be very different on this path than on normal city streets.

To explore the difference between this and other routes, I compared travel times for trips in two categories: first, trips that start and end at the 8 stations on the Hudson River Greenway and the adjacent avenue (ie 12th and 11th Avenues), and second, trips between these stations and 8 stations on the other side of the island.

I obtained data from Google Maps on the distance and predicted travel time between each pair of stations.
I used the [Google Maps distance matrix api](https://developers.google.com/maps/documentation/distance-matrix/intro) to obtain this data.
For the full project, I will probably need to pay some money to get a few more requests available, and batch requests as there is an 8 waypoint limit.
I would also like to look into using the full [Google Maps directions api](https://developers.google.com/maps/documentation/directions/intro), which includes full route results.
I would include this route information as a feature in my model, probably as a categorical variable based on the steets in the routes.
Route information could also be included as, say, a percentage of the route length.
This would require some in depth parsing of the route information.

Unfortunately, my results from this proof of concept do not indicate that this will be a straightforward project, and were generally counter to my expectations.
I had hypothesized that the two different categories of routes, those along the Hudson River Greenway (west_side in my code) and those that crossed Manhattan (crosstown in my code) would have different speeds.
In order to check this, I did a linear regression with three different variables present: the distance given for the Google maps route, a binary variable indicating whether the trip is a cross town trip or on the Hudson River Greenway, and a product of the two other variables.

The relationship between the predicted travel times and the route is different from what I had expected.
I had though that the Google Maps algorithm would be very simple: it expects bikes to travel at a constant speed, and thus travel times should be exactly linearly related to distance, with no consideration of speeds on the routes chosen.
Looking at the predicted travel times shown in the first plot, you can see that is not the case.
There seems to be some variation depending on the exact route, and there is a clear constant difference in travel times between the cross town routes and the routes along the Hudson River Greenway.

I expect that this constant difference may be from Google's model of traffic lights.
The variation in distance in the cross town trips comes from travel along the north south axis of the island, because of my choice of stations which are all along two north - south streets.
Because of light timing on north - south avenues is better aligned with travel, there is a low probability of hitting a red light for north - south travel, so it doesn't increase travel time.
Since the east west travel distance is constant, traffic lights appear to contribute a constant travel time, even though the contribution would generally scale with the number of traffic lights passed through, and thus roughly the distance traveled.


My hypothesis leads to a hypothesis test.
My hypothesis is that the speed of travel on crosstown routes is different from that on the Hudson River Greenway, so the product of the crosstown and distance variables should have a non-zero coefficient in the model.
The null hypothesis is that the product of the two variables does not factor into the data, and the product of the two variables has a coefficient of zero.
Preforming a t-test on the product variable, I find a fairly high probability of 16.5% of generating this data under the null hypothesis.
Even using a generous threshold of 5%, this does not allow me to reject the null hypothesis.
Even the hypothesis that there is a constant difference depending on the route was further rejected.
However, there does seem to be a very low probability that this data could be generated from a constant model, with a probability from an F-test of 8.44x10^-11 for the model that contains only an intercept and a linear relationship between the route distance.

The reasons for this hypothesis test result can be seen in the second plot. The data are very noisy.
The very long error bars for the 95% confidence intervals are because there is very large variance in travel time.
Citibike riders clearly don't simply ride directly between their starting and ending stations.
Some riders ride much slower than others, and some rides probably have other stops where the bike is not checked in at a station.
Many trips have a duration that is four times the mean.
While I may be able to reduce this noise by filtering out outliers, I imagine much of it may not be able to be eliminated.
As most tourists are "Customer" type users, one might expect that removing this user type would improve the variance, as tourists would be expected to ride slower and stop to take pictures, however they make up a fairly small portion of the sample and removing them does not make a huge difference.

Also, my choice of stations was not ideal.
the cross town trips are almost all longer than the Hudson River Greenway trips, so there is significant collinearity between the cross town and distance variables.

The data I choose may not be the most ideal.
The Hudson River Greenway is a popular route for cyclists looking to get exercise, who may have very different behavior from the largely commuter riders who are traveling cross town.

If it is possible to use bike share data to improve bicycling time estimates, it will require significant noise reduction.
However, using a full data set with many stations, along with data from other bike shares may work much better than my proof of concept.
