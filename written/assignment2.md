# Reading Assignment 2

[**When (ish) is My Bus?: User-centered Visualizations of Uncertainty in Everyday, Mobile Predictive Systems**](https://dl.acm.org/doi/10.1145/2858036.2858558)

0032763535 - Shuang Wu (wu1716)

## Summary

Prediction is not a single number, it is a probability distribution. However,
most people do not realize uncertainty is always present in predictions. Thus,
from time to time, users blame the system for suggesting a wrong predicted
result. This paper aims to provide better visualization of uncertainty in
predictive systems. The authors believe that the additional information about
uncertainty can help users to better understand the system and build trust in
the application.

The team first analyzed the existing literature on uncertainty visualization.
They found that many attempts tried to represent uncertainty with complex visual
representations of probability distributions. However, it would be difficult for
laypeople to grasp the meaning of the complicated graph. Also, according to some
studies, people tend to have a more accurate estimation of the probability of a
discrete outcome than a continuous one. However, numbering all the possible
outcomes is not feasible for many applications.

The writers show that Hypothetical Outcome Plots (HOP) are another way to
visualize uncertainty. However, since it involves long-time demonstration and
animated representation, it is not suitable for users to quickly get the idea.
Namely, it is not glanceable. The article also shows that the tradeoff between
glanceability and false precision is a common problem in data visualization. The
designer should try to find a balance between the two to make sure the chart is
appropriately glanceable and not misleading.

To overcome the problem mentioned above, the authors introduced a novel way to
convey uncertainty in predictive systems. They created quantile dotplots which
makes probability estimates more precise and easier to understand.

For the evaluation, the authors experimented to see how well each visualization
performs. First, they listed users' common goals when using a bus arrival time
prediction mobile app. Then, they asked the users about the unaddressed needs of
the app. This includes the following:

- Status probability: How likely is it that the bus will arrive on time, early,
  or delayed?
- Prediction variance: What is the chance the predicted results will be changed?
- Schedule frequency: How frequently do the buses arrive at various times of the
  day?

The authors then designed a set of visualizations to address the needs. The
requirements for the visualizations are:

- Simple text estimate of time to arrival
- Probabilistic ETA
- Probabilistic estimated arrival status
- Incorporate data freshness into the visualization

The experiment involved 541 participants and 6 different types of
visualizations, including:

- Density
- Stripeplot
- Density + Stripeplot
- Dotplot (20 quantiles)
- Dotplot (50 quantiles)
- Dotplot (100 quantiles)

The results show that the dotplot with 20 quantiles is the best visualization
for the scenario with few possible outcomes as it can take advantage of
subitizing. People can quickly estimate the probability of the bus arriving
status by counting the dots. Overall, dotplots are easier to use for general
audiences as they are more glanceable and more precise. Also, people prefer
density plot as it is more visually appealing than the dotplots. The reason
might be that the dotplots are visually busier and unfamiliar to the users. On
the other hand, stripeplot is the worst visualization according to the
experimental results. It has the highest error in estimated probability and the
largest standard deviation in it.

## Insights

I think providing the uncertainty information with probability distribution to
the user is a good starting point for many potential future works. For example,
a cross-domain service provider can combine the bus arrival probability with the
user's personal schedule to produce a deeply personalized action suggestion.
When a user is waiting for a bus with a navigation app, the app can analyze the
destination to analyze the importance of the next event. If the user is going to
an office, the app could suggest the user stay at the stop to avoid missing the
bus. Contrarily, if the user is going back home, the app could show a
notification showing some nearby grocery stores and suggest the user look around
before the bus arrives. If the bus comes early, it is not a big deal as the user
is not in a hurry. We can provide more customized suggestions to the user with
the uncertainty information, which could indicate a new opportunity for the
advertisement provider.

Apart from the above, since the probability of the bus status and arrival time
is provided to the users, we can also add some interactivity to show the factors
that affect the probability. For instance, the probability of the bus arriving
could be mainly based on the bus schedule and the historical record. However, if
all the users using the app while waiting for the bus at the previous stop have
suddenly stopped using the app, it is possible they are all on the bus
successfully, and thus the probability distribution of the ETA on the current
stop could be adjusted according to the number and timing of the riders taking
the bus at the previous stop. With the metadata, the distribution can be more
accurate and explainable to the users.

The article mentioned that some participants were afraid that they would be
responsible for the wrong decision if they are given uncertainty information so
they could not blame the system. I believe this is the most crucial reason for
all existing predictive systems to reveal such information to the target
audience. By giving the probability distribution of the prediction, there would
be less negative feedback from the users as they can understand the system and
take the responsibility for the derived decision. Minimizing bad feedback means
higher user satisfaction and a lower churn rate. Namely, the business owner
would benefit from the system with uncertainty information.
