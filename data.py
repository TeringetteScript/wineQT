def data_loading_cleaning(filename):
  #We are reading the data from a csv input
  Bor = pd.read_csv(filename)
  #Here we are dropping the NA data to clean the initial dataset
  Bor = Bor.dropna()

  #Collecting the features and attributes of the input data as a dictionary
  Bor = Bor.rename(columns={
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide"
})
  
  return Bor
