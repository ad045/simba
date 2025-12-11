try:
    import gnm
    print("gnm imported successfully")
    print(gnm.__file__)
    from gnm.fitting import RunConfig
    print("RunConfig imported")
    print(dir(RunConfig))
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
