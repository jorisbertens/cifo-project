![Git!](https://git-scm.com/images/logos/1color-orange-lightbg@2x.png)
## Git basics

__Get the whole repository(code) from the internet__
> git clone {repo_url}

__Get changes from the remote repository__
> git pull

__Add changes to files to be commited__
> git add {name of the file}
> git add -A (Adds all files changed)

__Commit changes__
> git commit -m "Commit message explaining the changes made"

__Send changes to the remote__
> git push

#### Git normal flow

1. git pull (so you start working on the latest version)
1. change the files and save
1. git add
2. git commit
3. git pull
  1. No conflit
    1. git push
  2. Conflict occured
    1. fix conflict
    2. git commit
    3. git push  

![Git workflow](https://camo.githubusercontent.com/2f9bc8ae52acf8f5bcf4217f4184fdcb0213eef6/68747470733a2f2f7777772e6769742d746f7765722e636f6d2f6c6561726e2f636f6e74656e742f30312d6769742f30312d65626f6f6b2f656e2f30312d636f6d6d616e642d6c696e652f30342d72656d6f74652d7265706f7369746f726965732f30312d696e74726f64756374696f6e2f62617369632d72656d6f74652d776f726b666c6f772e706e67)

Notes:  

*  Merge conflicts can be complex and should be dealt with carefully  
*  If the command line is not for you, I would recommend GitKraken as very good software to work with Git  
*  If you felt this short guide was not enough there are plenty of better guides online for git :)  
