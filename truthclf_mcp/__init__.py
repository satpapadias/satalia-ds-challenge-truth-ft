"""MCP server layer over truthclf.

Two servers, split on the credential boundary:

    model-tools   predict, explain          holds the provider credential, the
                                            response cache, the calibrators
    data-tools    dataset, metrics, retrieval   holds data.csv and a TF-IDF index,
                                            no provider credential

This package is the ONLY place that turns JSON into truthclf.data.Row objects.
truthclf itself is unmodified and takes dataclasses, as it always did.
"""

__all__ = ["adapter", "errors"]
