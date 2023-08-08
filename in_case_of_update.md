# In Case of Update
As `Propulate`* is under continuous development, from time to time there are updates on the `master` branch in the `Propulate` repo.

In order to be able to update `Propulate` into this repo, please do the following:

1.  If not present, create a branch `propulate` that points to the `Propulate` repo:
    ```
    git remote add propulate git@github.com:Helmholtz-AI-Energy/propulate.git
    ```
2.  Then, call
    ```
    git pull propulate master
    ```
    in order to pull the current state of `Propulate`'s repo's `master` branch into your own `master` branch.
3.  You will likely get a merge conflict which you can then resolve by using the built-in tools of PyCharm.
    If you are not using PyCharm please refer to the Git manual and look up how to resolve the conflicts.
4.  Afterwards, commit everything and push it onto your own GitHub repo so everything is fine.
