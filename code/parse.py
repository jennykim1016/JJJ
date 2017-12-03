"""
python_arXiv_parsing_example.py
This sample script illustrates a basic arXiv api call
followed by parsing of the results using the
feedparser python module.
Please see the documentation at
http://export.arxiv.org/api_help/docs/user-manual.html
for more information, or email the arXiv api
mailing list at arxiv-api@googlegroups.com.
urllib is included in the standard python library.
feedparser can be downloaded from http://feedparser.org/ .
Author: Julius B. Lucks
Contributor: Andrea Zonca (andreazonca.com)
This is free software.  Feel free to do what you want
with it, but please play nice with the arXiv API!
"""

import urllib
import feedparser

filestream = open('../data/label-abstract.txt', 'w')
# Base api query url
base_url = 'http://export.arxiv.org/api/query?';

max_results = 30
start = 0

categories = []
with open('../data/tags.txt') as f:
    tags = f.readlines()
    for tag in tags:
        categories.append(tag.split('\t')[0])

for category in categories:
    search_query = 'cat:' + category
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start, max_results)
    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

# perform a GET request using the base_url and query
    response = urllib.urlopen(base_url+query).read()

# change author -> contributors (because contributors is a list)
    response = response.replace('author','contributor')

# parse the response using feedparser
    feed = feedparser.parse(response)

# print out feed information
    #print 'Feed title: %s' % feed.feed.title
    #print 'Feed last updated: %s' % feed.feed.updated

# print opensearch metadata
    #print 'totalResults for this query: %s' % feed.feed.opensearch_totalresults
    #print 'itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage
    #print 'startIndex for this query: %s'   % feed.feed.opensearch_startindex

# Run through each entry, and print out information
    for entry in feed.entries:
        filestream.write('<id>')
        filestream.write('\n'+entry.id.split('/abs/')[-1])
        # primary category in the second line
        filestream.write('\n<category>')
        # Only (math.ST, stat.TH), (math-ph, math.MP), and (math.IT, cs.IT) are same
        # 0.61 precision, 0.55 recall, 0.54 f-score with combined.
        # 0.63 precision, 0.59 recall, 0.58 f-score with regular.
        #tags = [tag['term'] for tag in entry.tags]
        #if "math-ph" in tags:
        #    filestream.write('\nmath.MP')
        #elif "math.ST" in tags:
        #    filestream.write('\nstat.TH')
        #elif "math.IT" in tags:
        #    filestream.write('\ncs.IT')
        #else:
        #    filestream.write('\n'+entry.tags[0]['term'])
        filestream.write('\n'+entry.tags[0]['term'])
        filestream.write('\n<abstract>')
        filestream.write('\n'+entry.summary+'\n')
        filestream.write('\n')

filestream.close()
