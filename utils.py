import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.stats import diagnostic
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import requests
from bs4 import BeautifulSoup

def successive(values, many=2):
    return any(all(values[i+j] == "NAN" for j in range(many)) for i in range(len(values) - many + 1))

def select(lst, cut = 0.3):
  top = lst[:int(cut*len(lst))]
  bottom = lst[:-int(cut*len(lst))]
  return top, bottom

def clean_companies(df, many_return, many_category):

  df = df.fillna("NAN")
  drop_companies = []
  drop_col = []
  n_companies = int(len(df.columns[1:].copy())/2) + 1

  #return drop
  for col in df.columns[1:n_companies]:
    if successive(df[col], many_return):
      drop_companies.append(df.loc[0, col])
  #category drop
  for col in df.columns[n_companies:]:
    if successive(df[col], many_category):
      drop_companies.append(df.loc[0, col])

  drop_companies = list(set(drop_companies))
  for col in df.columns[1:]:
    if df.loc[0, col] in drop_companies:
      drop_col.append(col)
  df_clean = df.drop(drop_col, axis = 1)
  df_clean

  print("dropped companies with consecutive nans")

  df_clean = df_clean.replace(["NAN"], np.nan)
  df_clean.iloc[1,:] =  df_clean.iloc[1,:].fillna(method = "ffill")
  df_fill = df_clean.copy()
  print("forward fill of nan done")
  return df_fill

def generate_ratios(df, many_return=2, many_category=2, 
                    category = False, additional = False, rm_name = "something", version = 1):

  df = df.fillna("NAN")

  drop_companies = []
  drop_col = []
  n_companies = int(len(df.columns[1:].copy())/2) + 1

  #return drop
  for col in df.columns[1:n_companies]:
    if successive(df[col], many_return):
      drop_companies.append(df.loc[0, col])
  #category drop
  for col in df.columns[n_companies:]:
    if successive(df[col], many_category):
      drop_companies.append(df.loc[0, col])

  drop_companies = list(set(drop_companies))
  for col in df.columns[1:]:
    if df.loc[0, col] in drop_companies:
      drop_col.append(col)
  df_clean = df.drop(drop_col, axis = 1)
  df_clean

  print("Dropped companies with consecutive nans")

  df_clean = df_clean.replace(["NAN"], np.nan)
  if (version == 1):
    df_clean.iloc[1,:] =  df_clean.iloc[1,:].fillna(method = "ffill")
  if (version == 2):
    for vdx in range(20):
      df_clean.iloc[1,:] =  df_clean.iloc[1,:].fillna(method = "ffill")

  df_fill = df_clean.copy()
  print("Forward fill of NAN done \n")
  sector_list = pd.DataFrame(df_fill.columns[1:], columns = ["companies"])
  sector_list["companies"] = sector_list["companies"].apply(lambda c : c.split(".")[0])
  sector_list_pre = pd.DataFrame(df.columns[1:], columns = ["companies"])
  sector_list_pre["companies"] = sector_list_pre["companies"].apply(lambda c : c.split(".")[0])
  print("Companies removed:")
  companies_before = sector_list_pre.groupby(by = "companies").size()/2
  companies_after = sector_list.groupby(by = "companies").size()/2
  ##display(companies_before - companies_after)
  (companies_before - companies_after).to_csv(f"{rm_name}_companies_removed.csv")
  #df_fill
  sectors = sector_list["companies"].unique()

  data_ratios_mean = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
  data_ratios_std = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
  data_ratios_median = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
  data_ratios_q1 = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
  data_ratios_q3 = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
  data_ratios = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])

  print("Sector ranking and ratios")
  for sdx, s in enumerate(sectors):
    col_to_use = []

    for col in df_fill.columns:
      if col.split(".")[0] == s:
        col_to_use.append(col)


    ds = df_fill[col_to_use].copy()
    companies = ds.loc[0, :].values.tolist()
    ds.columns = companies
    ds = ds[1:].reset_index(drop = True)

    idx_cat = int(len(companies)/2)
    ds_cat = ds.iloc[:, idx_cat:].copy().applymap(lambda num: float(num))
    ds_return = ds.iloc[:, 0:idx_cat].copy().applymap(lambda num: float(num))

    rankings = []
    for row in range(len(ds_cat)):
      date_category = ds_cat.iloc[row, :].T
      date_category.sort_values(inplace = True, ascending = False)
      rankings.append(date_category.index.values.tolist())


    new_return = []
    new_category = []
    for row in range(len(ds_return)):
      date_return = ds_return[rankings[row]].iloc[row].values.tolist()
      date_category = ds_cat[rankings[row]].iloc[row].values.tolist()

      new_return.append(date_return)
      new_category.append(date_category)

    for ldx, lst in enumerate(new_return):
      top, bottom = select(lst)
      #mean
      data_ratios_mean.loc[ldx, s + "_top"] = np.mean(top)
      data_ratios_mean.loc[ldx, s + "_bottom"] = np.mean(bottom)
      #median
      data_ratios_median.loc[ldx, s + "_top"] = np.median(top)
      data_ratios_median.loc[ldx, s + "_bottom"] = np.median(bottom)
      #std
      data_ratios_std.loc[ldx, s + "_top"] = np.std(top)
      data_ratios_std.loc[ldx, s + "_bottom"] = np.std(bottom)
      #q1
      data_ratios_q1.loc[ldx, s + "_top"] = np.quantile(top, .25)
      data_ratios_q1.loc[ldx, s + "_bottom"] = np.quantile(bottom, .25)
      #q3
      data_ratios_q3.loc[ldx, s + "_top"] = np.quantile(top, .75)
      data_ratios_q3.loc[ldx, s + "_bottom"] = np.quantile(bottom, .75)
      #ratio
      data_ratios.loc[ldx, s] = np.mean(top)/np.mean(bottom)

    category_ratios_std = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
    category_ratios_mean = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
    category_ratios_median = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
    category_ratios_q1 = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
    category_ratios_q3 = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"])
    category_ratios = pd.DataFrame(df_clean.iloc[1:, 0].values.tolist(), columns = ["date"]) 

    for ldx, lst in enumerate(new_category):
      top, bottom = select(lst)
      #mean
      category_ratios_mean.loc[ldx, s + "_top"] = np.mean(top)
      category_ratios_mean.loc[ldx, s + "_bottom"] = np.mean(bottom)
      #median
      category_ratios_median.loc[ldx, s + "_top"] = np.median(top)
      category_ratios_median.loc[ldx, s + "_bottom"] = np.median(bottom)
      #std
      category_ratios_std.loc[ldx, s + "_top"] = np.std(top)
      category_ratios_std.loc[ldx, s + "_bottom"] = np.std(bottom)
      #q1
      category_ratios_q1.loc[ldx, s + "_top"] = np.quantile(top, .25)
      category_ratios_q1.loc[ldx, s + "_bottom"] = np.quantile(bottom, .25)
      #q3
      category_ratios_q3.loc[ldx, s + "_top"] = np.quantile(top, .75)
      category_ratios_q3.loc[ldx, s + "_bottom"] = np.quantile(bottom, .75)
      #ratio
      category_ratios.loc[ldx, s] = np.mean(top)/np.mean(bottom)
  


  if (category): 
    if (additional):
      return category_ratios, category_ratios_mean, category_ratios_median, category_ratios_std, category_ratios_q1, category_ratios_q3
    else:
      return category_ratios, category_ratios_mean
  if (additional):
    return data_ratios, data_ratios_mean, data_ratios_median, data_ratios_std, data_ratios_q1, data_ratios_q3
  else:
    return data_ratios, data_ratios_mean

def french_xl(df):
  return df.applymap(lambda val: float(".".join(val.split(","))) if type(val) != float else val)

def regression(df, policy, market_values, confounders, which = None, regplots = True, 
               descriptive_plots = True, model_results = True, coeff_plot = True, pearson = True):

  num_sectors = len(market_values)
  plot_rows = num_sectors/3 if num_sectors%3 == 0 else int(num_sectors/3) + 1

  print("The policy for regression is: ", policy)
  
  if (regplots):
    fig, ax = plt.subplots(plot_rows,3, figsize = (14, 14))
    sns.set_theme()
    for idx, snp in enumerate(market_values):
      sns.regplot(x = df[policy], y = df[snp], ax = ax[int(idx/3), idx % 3])
    plt.tight_layout()
    plt.show()

  #regression data
  reg_res = pd.DataFrame()

  for idx, snp in enumerate(market_values):
    if confounders:
      additional_model = "+".join(which)
      mod = smf.ols(formula=f'{snp} ~ {policy} + {additional_model}',
                  data=df)
    else:
      mod = smf.ols(formula=f'{snp} ~ {policy}',
                  data=df)
    res = mod.fit()
    reg_res.loc[idx, "Variables"] = snp
    reg_res.loc[idx, "Coefficient"] = res.params.loc[policy]
    reg_res.loc[idx, "P-value"] = res.pvalues.loc[policy]
    reg_res.loc[idx, "Coefficient STD"] = res.bse.loc[policy]
    if(model_results):
      print(res.summary())

  #plots and histograms
  if (descriptive_plots):
    fig, ax = plt.subplots(plot_rows,3, figsize = (14, 14))
    for idx, col in enumerate(market_values + [policy]):
      ax[int(idx/3), idx % 3].plot(df["time"], df[col])
      ax[int(idx/3), idx % 3].set_title(col)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(plot_rows,3, figsize = (14, 14))
    for idx, col in enumerate(market_values + [policy]):
      ax[int(idx/3), idx % 3].set_title(col)
      ax[int(idx/3), idx % 3].hist(df[col])
    plt.tight_layout()
    plt.show()

  #parameter ranking
  # standard errors
  variables = reg_res["Variables"]

  # coefficients
  coefficients = reg_res["Coefficient"]

  # p-values
  p_values = reg_res["P-value"]

  # standard errors
  standard_errors = reg_res["Coefficient STD"]

  if (coeff_plot):
    l1, l2, l3, l4 = zip(*sorted(zip(coefficients, variables, standard_errors, p_values)))
    # Set Seaborn style

    # Create the error bar plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(l1, np.arange(len(l1)), xerr=2 * np.array(l3), linewidth=1,
                linestyle='none', marker='o', markersize=6,
                markerfacecolor='royalblue', markeredgecolor='black', capsize=5)

    plt.vlines(0, -0.5, len(l1) - 0.5, linestyle='--', color='gray', alpha=0.7)  # Use gray color for the vertical line
    for pdx, pval in enumerate(list(l4)):
      if (pval <= 0.2 and pval > 0.1):
        plt.scatter(l1[pdx], pdx + 0.4, marker='x', s = 50, color = "k")
      if (pval <= 0.1 and pval > 0.05):
        plt.scatter(l1[pdx], pdx + 0.4, marker='X', s = 50, color = "k")
      if (pval <= 0.05 and pval > 0.01):
        plt.scatter(l1[pdx], pdx + 0.4, marker='*', s = 50, color = "k")
      if (pval <= 0.01):
        plt.scatter(l1[pdx], pdx + 0.4, marker='*', s = 50, color = "gold")
      
    # Customize plot aesthetics
    plt.xticks(fontsize=9)
    plt.yticks(np.arange(len(l2)), l2, fontsize=12)

    plt.xlabel('Coefficients', fontsize=12, fontweight='bold')
    plt.ylabel('Variables', fontsize=12, fontweight='bold')
    plt.title(f'{policy} Coefficients per S&P500 Sector', fontsize=14, fontweight='bold')

    # Remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Add horizontal grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Ensure all elements are visible
    plt.savefig(f"plots/errorbars_{policy}.jpg",dpi=300, bbox_inches='tight')
    plt.show()

  reg_res.to_csv(f"plots/{policy}_regression_results.csv")
  #pearson correlation
  pearson_df = pd.DataFrame()
  if (pearson):
    for idx, snp in enumerate(market_values):
      corr, pval = pearsonr(df[policy].fillna(0), df[snp].fillna(0))
      pearson_df.loc[snp, "Correlation"] = corr
      pearson_df.loc[snp, "P-value"] = pval

    pearson_df = pearson_df.sort_values("Correlation")

  display(reg_res.sort_values(by = "Coefficient", ascending= False), pearson_df)
  print("success")
  return 0

def add_time(df, Xformat = False):
  if (Xformat): 
    month_keys = list(df.iloc[:, 0].apply(lambda d: d.split("-")[0]).unique())
    month_values = list(range(1, 13))
    month_dictionary = {month_keys[i]: month_values[i] for i in range(len(month_keys))}
    df["Month"] = df.iloc[:, 0].apply(lambda d: month_dictionary[d.split("-")[0]])
    df["Year"] = df.iloc[:, 0].apply(lambda d: int("20" + d.split("-")[1]))

  df["time"] = df.apply(lambda d: round(d["Year"] + (d["Month"]-0.5)/12, 2), axis = 1)
  return df

def add_cols(df1, df2, cols):
  return df1.merge(right=df2[cols], left_on = "time", right_on = "time")

def get_company_full_name(stock_name):
    url = f"https://www.reuters.com/markets/companies/{stock_name}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        company_name_element = soup.find("h1", attrs={"data-testid": "Heading", "class": "markets-header__company-name__gDs1M"})
        if company_name_element:
            company_name = company_name_element.text.strip()
            return company_name
        else:
            print("Could not find the company information.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def company_regression(category, reg_df, reg_cols, policy, many_return = 2, many_category = 2,
                        confounders = False, which = [""], split_2 = False, manhattan = False, split_significant = False,
                          suggested = 1.5, real_names = False, adaptive = False):
  df = pd.read_csv(f"data_fix/NCloseX{category}.csv")
  df_c = clean_companies(df, many_return=many_return, many_category=many_category)
  company_sector = pd.Series([(df_c.iloc[0, 1:].values.tolist()[i], pd.Series(df_c.columns[1:].
                              values.tolist()).apply(lambda sc: sc.split(".")[0]).values.tolist()[i])
                                for i in range(df_c.shape[1] - 1)]).unique()
  #display(df_c.sample())
  company_sector = {t[0] : t[1] for t in company_sector}
  ds = df_c.iloc[1:, :].copy()
  ds.columns = df_c.iloc[0, :]
  ds.columns = pd.Series(ds.columns).apply(lambda col: "_".join(col.split(".")) if (type(col) == str) else col)

  ds_cat = ds.iloc[:, len(company_sector) + 1:]
  ds_return = ds.iloc[:, 1:len(company_sector)]
  ds = add_time(ds, Xformat=True)
  ds_return["time"] = ds["time"].copy()
  ds_cat["time"] = ds["time"].copy()
  ds_cat = ds_cat.applymap(lambda num: float(num))
  ds_return = add_cols(ds_return, reg_df, reg_cols).applymap(lambda el: float(el))

  #regression
  reg_res = pd.DataFrame()
  for cdx, company in enumerate(ds_return.columns[:len(company_sector)-1]):
    #print(ds_return[policy], ds_return[company])
    if confounders:
      additional_model = "+".join(which)
      mod = smf.ols(formula=f'{company} ~ {policy} + {additional_model}',
                  data=ds_return)
    else:
      mod = smf.ols(formula=f'{company} ~ {policy}',
                  data=ds_return)
    res = mod.fit()
    reg_res.loc[cdx, "Company"] = company
    reg_res.loc[cdx, "beta_coef"] = res.params.loc[policy]
    reg_res.loc[cdx, "P-value"] = res.pvalues.loc[policy]

  #Descriptive statistics
  print("regressions done")
  reg_res["sector"] = reg_res["Company"].apply(lambda co: company_sector[".".join(co.split("_"))])
  reg_res["position_man"] = np.random.uniform(size=len(reg_res), low=0, high=2)
  reg_res["pvalue_man"] = -np.log10(reg_res["P-value"])

  reg_res[f"L2Y Average {category}"] = reg_res["Company"].apply(lambda co:
                      ds_cat.query(f"time > {str(ds_cat.time.max() - 2)}")[co].mean())
  
  if(manhattan):
    plt.scatter(reg_res.position_man, reg_res["pvalue_man"], alpha = 0.7)
    alpha = 0.05
    bonferroni_logged = -np.log10(alpha/len(reg_res))
    plt.axhline(y = -np.log10(alpha), color = 'r', linestyle = "-.", label = r"$\alpha = 0.05$")
    plt.axhline(y = bonferroni_logged, color = 'g', linestyle = "--", 
            label = "Bonferroni correction significance")
    plt.legend()
    plt.show()
    print(f"Companies the most impacted by {policy} with suggested significance level of 2 (in -log10):")
    if (real_names):
      good_companies = reg_res.query("pvalue_man > 2")
      good_companies["full_name"] = good_companies["Company"].apply(lambda co:
                                                                    get_company_full_name(".".join(co.split("_"))))
      display(good_companies)
  
  if (split_significant):
    print("\n Filtering out non signficiant company regression coefficients")
    reg_res = reg_res.query(f"pvalue_man > {suggested}")
  
  if (split_2):

    #L2Y = Last 2 year
    positive = reg_res[reg_res["beta_coef"] > 0]
    negative = reg_res[reg_res["beta_coef"] <= 0]
    print("\n Positive company count:", len(positive), "|| Negative company count:", len(negative))

    if (not len(negative) and adaptive): 
      print(" Changing negative companies to companies below average coefficient\n")
      coef_average = reg_res["beta_coef"].median()
      positive = reg_res[reg_res["beta_coef"] >= coef_average]
      negative = reg_res[reg_res["beta_coef"] < coef_average]
    sector_count_positive = positive.groupby("sector").count().sort_values(by="Company")
    sector_count_negative = negative.groupby("sector").count().sort_values(by="Company")
    fig, ax = plt.subplots(2,1, figsize = (10, 9))
    ax[0].set_xticklabels(ax[0].get_xticks(), rotation = 45)
    ax[0].set_title(f"Positive Regression Coefficient with {policy}")
    sns.barplot(x = sector_count_positive.index, y = sector_count_positive["Company"].values, ax = ax[0], palette='husl')
    
    ax[1].set_xticklabels(ax[1].get_xticks(), rotation = 45)
    ax[1].set_title(f"Negative Regression Coefficient with {policy}")
    if (len(negative)):
      sns.barplot(x = sector_count_negative.index, y = sector_count_negative["Company"].values, ax = ax[1], palette='husl')
    plt.tight_layout()
    print(f"L2Y Average {category} of positive correlated companies:", positive[f"L2Y Average {category}"].mean())
    print(f"L2Y Average {category} of negative correlated companies:", negative[f"L2Y Average {category}"].mean())
    print("student's t-test results:", ttest_ind(positive[f"L2Y Average {category}"], 
                                                 negative[f"L2Y Average {category}"]))
    plt.show()
    plt.hist(reg_res["beta_coef"])
    plt.show()

  return "success"