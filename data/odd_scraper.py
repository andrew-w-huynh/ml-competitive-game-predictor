from bs4 import BeautifulSoup
import pandas as pd
import time, random
import requests, json
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# getting the money-line and spread odds for the entire season
def get_season_odds(date):
    # get the money-line odds
    page = browser.new_page()
    try:
        page.goto(f"https://www.oddstrader.com/nba/?date={date}&g=game&m=money")

        # uncomment the line below to get all lines
        # page.wait_for_selector("button[class^='buttons__Plain-sc-rlvloz-0 Oddstyles__Button-sc-rabb5n-1 lnKGbX jjXoGA']", timeout=15_000)

        # get just the best line
        page.wait_for_selector("span[class^='best-line']", timeout=15_000)
        html = page.content()
    except PlaywrightTimeoutError:
        print(f"[!] Timeout on {date}, skipping.")
        return
    finally:
        page.close()

    # parse the HTML content
    soup = BeautifulSoup(html, "html.parser")

    # get teams
    teams = soup.find_all('span', class_='teamName')
    for team in teams:
        team_name = team.get_text(strip=True)
        df.loc[len(df), "team"] = team_name

    # get scores for games
    offset = len(teams)
    scores = soup.find_all('div', class_='TeamContainerstyles__ConsensusOrScore-sc-y43dva-3')
    for score in scores:
        score_value = score.get_text(strip=True)
        df.loc[len(df) - offset, "score"] = score_value
        offset -= 1

    # get the money-line odds
    offset = len(teams)
    # uncomment the lines below to get all lines
    # column_num = 1
    # odds = soup.find_all('button', class_='buttons__Plain-sc-rlvloz-0 Oddstyles__Button-sc-rabb5n-1 lnKGbX jjXoGA')
    odds = soup.find_all('span', class_='best-line')
    for odd in odds:
        odd_value = odd.get_text(strip=True)
        df.loc[len(df) - offset, "best_ml_odds"] = odd_value
        offset -= 1

        # uncomment the lines below to get all lines
        # if column_num > 5:
        #     offset -= 1
        #     column_num = 1
        # df.loc[len(df) - offset, f"ml_odds_{column_num}"] = odd_value
        # column_num += 1

    # get the spread odds
    page = browser.new_page()
    try:
        page.goto(f"https://www.oddstrader.com/nba/?date={date}&g=game&m=spread")
        # uncomment the line below to get all lines
        # page.wait_for_selector("button[class^='buttons__Plain-sc-rlvloz-0 Oddstyles__Button-sc-rabb5n-1 lnKGbX jjXoGA']", timeout=15_000)

        # get just the best line
        page.wait_for_selector("span[class^='best-line']", timeout=15_000)
        html = page.content()
    except PlaywrightTimeoutError:
        print(f"[!] Timeout on {date}, skipping.")
        return
    finally:
        page.close()

    # get the spread odds
    soup = BeautifulSoup(html, "html.parser")
    offset = len(teams)
    # uncomment the lines below to get all lines
    # column_num = 1
    # odds = soup.find_all('button', class_='buttons__Plain-sc-rlvloz-0 Oddstyles__Button-sc-rabb5n-1 lnKGbX jjXoGA')
    odds = soup.find_all('span', class_='best-line')
    for odd in odds:
        odd_value = odd.get_text(strip=True)
        df.loc[len(df) - offset, "best_spread_odds"] = odd_value
        offset -= 1

        # uncomment the lines below to get all lines
        # if column_num > 5:
        #     offset -= 1
        #     column_num = 1
        # df.loc[len(df) - offset, f"spread_odds_{column_num}"] = odd_value
        # column_num += 1

# create a dataframe to store the all lines
# uncomment the line below if you want to get all lines
# df = pd.DataFrame(columns=["team", "score", "ml_odds_1", "ml_odds_2", "ml_odds_3", "ml_odds_4", "ml_odds_5", 
#                            "spread_odds_1", "spread_odds_2", "spread_odds_3", "spread_odds_4", "spread_odds_5"])

# create a datafram to store just the best lines
df = pd.DataFrame(columns=["team", "score", "best_ml_odds", "best_spread_odds"])

# hardcoding the first day of each month to prevent detection of botting
# first day of regular season
# year, month, day = 2024, 10, 22
# year, month, day = 2024, 11, 1
# year, month, day = 2024, 12, 1
# year, month, day = 2025, 1, 1
# year, month, day = 2025, 2, 1
# year, month, day = 2025, 3, 1
# year, month, day = 2025, 4, 1

# just the playoffs
# year, month, day = 2025, 4, 19
year, month, day = 2025, 5, 1
date = f"{year}{month:02d}{day:02d}"

# hardcoding last day of each month to prevent dectection of botting
# current_date = "20241032"
# current_date = "20241131"
# current_date = "20241232"
# current_date = "20250132"
# current_date = "20250229"
#current_date = "20250332"
# current_date = "20250431"
current_date = "20250511"

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    # scrape the data for each day of the season
    while date != current_date:
        print(date)
        get_season_odds(date)
        day += 1
        date = f"{year}{month:02d}{day:02d}"
        time.sleep(random.uniform(1.5, 4.0))
    
    browser.close()

# save the data to a CSV file
df.to_csv(f"odds/{month}_{year}_odds.csv", index=False)