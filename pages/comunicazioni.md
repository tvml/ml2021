---
layout: page
format: blog-index
# Don't index these pages dear Google.
noindex: true
title: "Comunicazioni"
#subheadline: "Informazioni"
teaser: ""
permalink: /comunicazioni/
header: no
---

<posts>
        <table>
            {% for post in site.categories.comunicazioni reversed %}
            <tr>
                <td><i class="icon-clock"></i> <time datetime="{{post.date}}">{{post.date|date:"%d-%m-%Y"}}</time></td>  
                <td><a href="{{site.baseurl}}{{ post.url }}">{{ post.title }}</a></td>
                <!--td><span class="category"><i class="icon-tag"></i> {{post.categories | category_links}}</span></td-->
            </tr>
            <tr>
                <td colspan='2'>
                {{ post.excerpt }}
                </td>
            </tr>
    	{% endfor %}
        </table>
</posts>



