---
layout: post
title:  "Starter Code for Quick Web Prototype"
author: ernestlwt
categories: [jekyll]
tags: [red, yellow]
image: assets/images/starter-pack.jpg
description: 'React+Flask+Postgres+Docker'
featured: true
hidden: true
rating: 5
---
Need to build a web prototype but no idea where to start? No worries! Ernest is here to the rescue. This post will provide you with some starter code using ~~SOTA~~ popular frameworks which you should have no problem finding stackoverflow answers to your questions.

# Disclaimer
I am no expert, but this starter code should be easy to use for both development and deployment. I intend to improve the [github repo](https://github.com/ernestlwt/prototype_starter_pack) as I gain more knowledge on this. So click here for more, click here to subscribe and dont forget to click the notification bell.

# About the Repo
You can find most of the information on the readme of the repository already. So really, there is not much for me to go through that is not repeating the readme on the repo. There are instructions on both running the servers during development and deployment.

The folder structure are just something I would use. Go ahead and organize as you like. If you find a better way to structure organize the folders, do let me know! It would be helpful to me and to others.

# Tips on Reactjs
Reactjs has gone through many changes. When I first started learning, I followed the popular tutorials which does not equip you with the new features of react. So below are some of the tip I would have wish someone told me before I started learning React.

### Functional Components vs Class Components

```
// class components
class Person extends Component {
    ...
}
```
```
// functional components
function Person(){
    ...
}
```
Whats the difference? Well *not so* recently they introduce [react hooks](https://reactjs.org/docs/hooks-intro.html), which is really powerful and it is only available for functional components. So if you are just starting out, I would highly recommend you do things the *functional components* way. So far, I have not met anything that I could not achieve with functional components and would require class components.

### React Routing
React is a Single Page Application(SPA). This means that there is just one html file, irregardless of how many different url you have, and javascript does the rest of the work to change the look of the page. If you have created multiple url, make sure you use some form of routing features instead of *\<a href\>* as you would lose the state of the web application(you will understand this as you continue learning react). There are many different routing packages but the one used in the starter code is react-router and you can find the tutorial [here](https://reactrouter.com/web/guides/quick-start).

### No jquery
If you have done any web application before, you would probably have used jquery before. JQuery allows you to change the webapp without reloading the page, however, react does this too! Using them both at the same time could result in unpredictable results. So no jquery for you. React should be able to do whatever you need

# Tips on Flask
Flask is pretty straight forward. It is design to be light-weight and you add extensions as you develop instead of them providing everything to you in the case of django. I prefer flask but your usage might vary.

### Structure? No Structure!
As mentioned, flask is meant to be lightweight and no folder structure is provided by default. For micro-services doing just a single task, you might not even have more than 100 lines of code, creating a folder structure might just be a waste of time. However when you start adding features to your prototype, you might start to scratch you head on how to organize your code. Thus, I have included a recommended folder structure.

### Use Gunicorn and Nginx
The server that comes with flask when you `flask run` is only for development. You might face issues using it during deployment when the number of requests/second gets higher etc. Gunicorn is a server that can serve your flask python code and it is production grade so use that instead. However, it might not be the most efficient for static files. No worries, you always have Nginx to do that for you. In fact, the react codes are compiled into static files and served using a docker container with Nginx, and the docker container for the backend server is already using Gunicorn!

# Conclusion
Once again, I am no expert, so if you ever get to use this starter code, please let me know what you think! I intend to continue improving this article and the repo.

Github link: https://github.com/ernestlwt/prototype_starter_pack

##### PS: please read the readme



