
## argument completion (for test script)

The testing script test.py uses argcomplete, so if you want to have argument completion please run once

    activate-global-python-argcomplete --dest=- >> ~/.bash_completion

afterwards you should be able to use completion with [Tab]

    ./test.py [Tab]
