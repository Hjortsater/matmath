def alias(*names):
    """
    ALLOWS FOR CLEANER IDENTITY ASSIGNMENT OF CLASS METHODS.
    """
    class AliasDescriptor:
        def __init__(self, func):
            self.func = func
            self.names = names

        def __set_name__(self, owner, name):
            for alias_name in self.names:
                setattr(owner, alias_name, self.func)

            setattr(owner, name, self.func)

        def __get__(self, instance, owner):
            return self.func.__get__(instance, owner)

    return AliasDescriptor