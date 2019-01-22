def get_advice(pandas_source, spark_target):
    msg = ""
    if pandas_source is not None and spark_target is not None:
        msg += """You are a Pro! You are trying to use pandas function {}, use spark function
        {}""".format(pandas_source, spark_target)
    elif pandas_source is not None and spark_target is None:
        msg += """You are a Pro! You are trying to use pandas function {}, checkout the spark
        user guide to find a relevant function""".format(pandas_source)
    elif pandas_source is None and spark_target is not None:
        msg += """You are a Pro! Use spark function {}""".format(spark_target)
    else:   # both none
        msg += """Checkout the spark user guide to find a relevant function"""
    return msg


class PandorableSparkyNotImplementedError(NotImplementedError):

    def __init__(self, pandas_source=None, spark_target=None, description=""):
        self.pandas_source = pandas_source
        self.spark_target = spark_target
        advisory = get_advice(pandas_source, spark_target)
        if len(description) > 0:
            description += " " + advisory
        else:
            description = advisory
        super(PandorableSparkyNotImplementedError, self).__init__(description)
