---
layout: page
title: Lottery Analytics
---

Inspired by [Dr. Richard Jarecki](https://thehustle.co/professor-who-beat-roulette) and as a way to refresh my data science skills, I conducted an investigation into our local GAA's lottery draw.

The weekly lotto results are available online and look like below (some information has been redacted to protect personal identities).

   <p style="text-align: center;">
     <img src="/assets/img/Lotto_eg.png" alt="Website example" style="max-width: 100%; height: auto;" />
   </p> 

First, I used [Selenium](https://www.selenium.dev/) to automate web browsing and locate the necessary pages for scraping. Then, I utilised BeautifulSoup to parse the web pages. By employing regex and Python, I extracted the relevant information and stored the results of 450 lotto draws.

   <p style="text-align: center;">
     <img src="/assets/img/results_no.png" alt="Results" style="max-width: 100%; height: auto;" />
   </p> 

Finally, I performed statistical analysis to determine the fairness of the lottery draw. This involved generating 10,000 and 450 samples from a uniform distribution and applying a chi-square test to assess the fairness of the game. The results of this analysis suggest the lotto is a fair game.

   <p style="text-align: center;">
     <img src="/assets/img/450_uni_gen.png" alt="450 Generated Results" style="max-width: 100%; height: auto;" />
   </p> 

  <p style="text-align: center;">
     <img src="/assets/img/10000_uni_gen.png" alt="10000 Generated Results" style="max-width: 100%; height: auto;" />
   </p> 

  <p style="text-align: center;">
     <img src="/assets/img/chisquare.png" alt="Chi-Square Results" style="max-width: 100%; height: auto;" />
   </p> 

Additionally, I gathered data on the winning numbers and prize amounts from previous draws, as well as identifying the numbers with the highest and lowest frequencies. 

  <p style="text-align: center;">
     <img src="/assets/img/winning_no.png" alt="Winning Numbers" style="max-width: 100%; height: auto;" />
   </p> 

  <p style="text-align: center;">
     <img src="/assets/img/high_low.png" alt="High-Low Numbers" style="max-width: 100%; height: auto;" />
   </p> 


Unfortunately even with this data-driven approach I still haven't won! 
