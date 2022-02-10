# homophily-outbreaks

This repository contains the source code for the two models used in the study [Effect of homophily and correlation of beliefs on COVID-19 and general infectious disease outbreaks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260973).

The first model (simple_network_model02published.py) models the spread of a generic infectious disease through a simple interaction network of individuals (nodes), where each individual possesses attributes that describe its vaccination status and its activity level (due to belief in social distancing measures). These node attributes may be correlated and people of similar beliefs may cluster (homophily).

The second model (ABM_network_model31published.py) is an extension of a previously published COVID model with the same node attributes as the first model, plus an additional third attribute corresponding to risk-status (high-risk vs low-risk individuals).
