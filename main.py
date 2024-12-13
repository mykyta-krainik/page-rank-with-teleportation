from pyspark import SparkContext, SparkConf
from random import randint


def parse_links(line):
    parts = line.split(" ")
    return parts[0], parts[1:]


def compute_contributions(urls, rank):
    num_urls = len(urls)
    return [(url, rank / num_urls) for url in urls]


def pagerank(input_path, output_path, num_iterations=10, damping_factor=0.85, threshold=1e-4):
    conf = SparkConf().setAppName("PageRank").setMaster("local")
    sc = SparkContext(conf=conf)

    lines = sc.textFile(input_path)

    links = lines.map(lambda line: parse_links(line)).cache()

    num_pages = links.count()
    ranks = links.mapValues(lambda _: 1.0 / num_pages)

    convergence_iterations = 0

    for iteration in range(num_iterations):
        convergence_iterations += 1
        contributions = links.join(ranks).flatMap(
            lambda url_urls_rank: [(url_urls_rank[0], 1.0 / num_pages)] + compute_contributions(url_urls_rank[1][0], url_urls_rank[1][1])
        )

        new_ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(
            lambda rank: damping_factor * rank + (1 - damping_factor) / num_pages
        )

        delta = new_ranks.join(ranks).mapValues(lambda x: abs(x[0] - x[1])).values().sum()
        ranks = new_ranks

        if delta < threshold:
            break

    print("Converged after %d iterations" % convergence_iterations)

    ranks_sum = ranks.values().sum()

    print("Sum of ranks: %f" % ranks_sum)

    ranks = ranks.mapValues(lambda rank: round(rank / ranks_sum, 4))

    ranks.coalesce(1).saveAsTextFile(output_path)

    sc.stop()


output_path = "output-%d" % randint(0, 10000)

pagerank("graph.txt", output_path, num_iterations=40, damping_factor=0.85)
