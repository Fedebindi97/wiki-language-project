# wiki-language-project
Inferring the network of bilingualism from Wikipedia data.

## Project overview
We will scrape a random subset of articles from the main Wikipedias, extracting the languages each article appears in. Then, we will construct a network where each node is a language and the edges are given by the number of articles appearing in both languages.

We will sample 0.3% of Wikipedia articles (for computational reasons), only consider the biggest 20 Wikis by number of articles, exclude the Wikipedias in Swedish, Cebuano and Waray (for extensive use of bots), and exclude articles appearing in more than 5 languages (to mitigate the pivot effect of English).

## Privacy
All Wikipedia text is available under the Creative Commons Attribution-ShareAlike License (CC-BY-SA).

## Usability of project materials
All project materials are free to use and reproduce without asking for author's consent. However, in case of use, at least a link to this repo shall be inserted (hey, I deserve it).