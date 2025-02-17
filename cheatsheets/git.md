git add .  #将当前目录下所有新文件和修改过的文件添加到暂存区
git status  #查看工作目录和暂存区的状态，显示未跟踪、已修改或已暂存的文件
git commit -m "xxx"   #提交暂存区的更改到本地仓库，提交信息为xxx
git push learning-repository main  #将本地main分支的更改推送到远程仓库learning-repository

git fetch learning-repository main  #从远程仓库拉取最新更改，但不合并
git merge learning-repository/main  #将远程分支合并到当前分支

git pull learning-repository main  #从远程仓库learning-repository拉取main分支的更改并合并到当前分支