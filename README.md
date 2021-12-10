# custom-nba-mlp-cpp

An implementation of https://github.com/imathur1/custom-nba-mlp in C++.

To build our project run `make exec`. Then run the `driver.cc` file by typing `./bin/exec`, which will read the college basketball player data from the CSV file into a DataFrame object.
Subsequently, an MLP model will be trained using DataFrame's data. After the model is trained you can predict the probability of a current college basketball player being drafted into the NBA using the `MLP.Predict()` function.
Many of these college basketball players are in the `prediction.csv` file, so you can predict their chances.
But even beyond the upcoming draft class, this model can be used for players of any draft class in the future.
To run our test suite, run `make tests` and then `./bin/tests`.
