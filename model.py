def choose_best_ccp_alpha(X, y, cv_splits=5, scoring='accuracy'):
    """
    Determine the best ccp_alpha for cost-complexity pruning.
    Returns:
        best_alpha_min_error : alpha giving minimum CV error
        best_alpha_1se       : alpha chosen by 1-SE rule (simplest tree within 1 SD of minimum)
        results              : dict with alphas, mean errors, std errors
    """

    # 1. Fit initial tree to get pruning path
    base_tree = DecisionTreeClassifier(random_state=0)
    path = base_tree.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas[:-1]      # drop trivial last alpha

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)
    mean_err = []
    std_err = []

    # 2. Evaluate CV error for each alpha
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
        scores = cross_val_score(tree, X, y, scoring=scoring, cv=cv)
        errors = 1 - scores                       # error = 1 - accuracy
        mean_err.append(errors.mean())
        std_err.append(errors.std(ddof=1))

    mean_err = np.array(mean_err)
    std_err = np.array(std_err)

    # 3. Find alpha with minimum CV error
    min_idx = np.argmin(mean_err)
    best_alpha_min_error = ccp_alphas[min_idx]

    # 4. Apply 1-SE rule (simplest tree within one std of the minimum)
    threshold = mean_err[min_idx] + std_err[min_idx]
    valid = ccp_alphas[ mean_err <= threshold ]
    best_alpha_1se = valid.max()   # simplest = largest alpha

    # 5. Package results
    results = {
        "ccp_alphas": ccp_alphas,
        "mean_error": mean_err,
        "std_error": std_err
    }

    return best_alpha_min_error, best_alpha_1se, results


def models(BOR):
  
  SEED = 42

  random.seed(SEED)
  np.random.seed(SEED)

  y = BOR["quality"]
  X = BOR.drop(columns=["quality"])

  # 3. Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)
  # 4. LASSO regresszió pipeline (standardizálás + LassoCV)
  lasso_pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("lasso", LassoCV(cv=10, random_state=42))
  ])

  # 5. Modell tanítása
  lasso_pipe.fit(X_train, y_train)

  # 6. Optimális lambda (alpha)
  optimal_alpha = lasso_pipe.named_steps["lasso"].alpha_
  print("Optimal alpha (λ):", optimal_alpha)

  # 7. Koeficiensk
  coefficients = pd.Series(
      lasso_pipe.named_steps["lasso"].coef_,
      index=X.columns
  )

  print(coefficients)


  Bor["Besorolas"] = np.where(Bor["quality"] < 5.7, "Rossz", "Jó")

  plt.hist(Bor["quality"], bins=15)
  #Baseline model
  plt.axvline(Bor["quality"].mean(), color='red', linestyle='--')
  plt.title("Értékelések eloszlása")
  plt.show()

  X = Bor.drop(columns=["Id", "quality", "Besorolas"])
  y = Bor["Besorolas"]

  X = pd.get_dummies(X, drop_first=True)
  #Train Test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)
  #Decision Tree
  dt = DecisionTreeClassifier(random_state=32, ccp_alpha=0.01)
  dt.fit(X_train, y_train)

  plt.figure(figsize=(12, 8))
  plot_tree(dt, filled=True, feature_names=X.columns)
  plt.show()

  dt = DecisionTreeClassifier(random_state=32, ccp_alpha=0.01)
  dt.fit(X_test, y_test)

  plt.figure(figsize=(12, 8))
  plot_tree(dt, filled=True, feature_names=X.columns)
  plt.show()

  path = dt.cost_complexity_pruning_path(X_train, y_train)
  ccp_alphas = path.ccp_alphas

  dt_best = DecisionTreeClassifier(random_state=32, ccp_alpha=ccp_alphas[10])
  dt_best.fit(X_train, y_train)

  rf = RandomForestClassifier(random_state=32)
  
  
  best_min, best_1se, res = choose_best_ccp_alpha(X_train, y_train)

  print("Best alpha (min CV error):", best_min)
  print("Best alpha (1-SE rule, preferred):", best_1se)
  

  return X_train, X_test, y_train, y_test, rf, dt

def features_and_fitting(train_test_rf_df):
  X_train, X_test, y_train, y_test, rf, dt=train_test_rf_df[0],train_test_rf_df[1],train_test_rf_df[2],train_test_rf_df[3],train_test_rf_df[4],train_test_rf_df[5]
  rf.fit(X_train, y_train)

  plt.plot(rf.feature_importances_)
  plt.title("Változó fontosság")
  plt.show()

  rf.fit(X_test, y_test)

  plt.plot(rf.feature_importances_)
  plt.title("Változó fontosság")
  plt.show()

  scores_dt = cross_val_score(dt, X_test, y_test, cv=10)
  scores_rf = cross_val_score(rf, X_test, y_test, cv=10)

  print("Döntési fa CV:", scores_dt.mean())
  print("Random forest CV:", scores_rf.mean())
  
  return scores_dt, scores_rf
