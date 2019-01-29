"""
Exceptions/Errors used in pandorable_sparky.
"""


def code_change_hint(pandas_function, spark_target_function):
    if pandas_function is not None and spark_target_function is not None:
        return "You are trying to use pandas function {}, use spark function {}" \
               .format(pandas_function, spark_target_function)
    elif pandas_function is not None and spark_target_function is None:
        return ("You are trying to use pandas function {}, checkout the spark "
                "user guide to find a relevant function").format(pandas_function)
    elif pandas_function is None and spark_target_function is not None:
        return "Use spark function {}".format(spark_target_function)
    else:   # both none
        return "Checkout the spark user guide to find a relevant function"


class SparkPandasNotImplementedError(NotImplementedError):

    def __init__(self, pandas_function=None, spark_target_function=None, description=""):
        self.pandas_source = pandas_function
        self.spark_target = spark_target_function
        hint = code_change_hint(pandas_function, spark_target_function)
        if len(description) > 0:
            description += " " + hint
        else:
            description = hint
        super(SparkPandasNotImplementedError, self).__init__(description)
