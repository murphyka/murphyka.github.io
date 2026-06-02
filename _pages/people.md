---
layout: page
title: People
permalink: /people/
nav: true
nav_order: 3
---

<style>
.people-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  margin: 1.5rem 0 2.5rem;
}
.person-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 150px;
  text-align: center;
}
.person-card img {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 50%;
}
.person-card .person-placeholder {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  color: #999;
}
.person-card .person-name {
  margin-top: 0.6rem;
  font-weight: 600;
  font-size: 0.95rem;
}
.person-card .person-bio {
  font-size: 0.82rem;
  color: #666;
  margin-top: 0.2rem;
}
.alumni-list {
  list-style: none;
  padding: 0;
  margin: 1rem 0 2.5rem;
}
.alumni-list li {
  padding: 0.25rem 0;
  font-size: 0.95rem;
}
.alumni-list .alumni-note {
  color: #666;
  font-size: 0.85rem;
  margin-left: 0.5rem;
}
</style>

---

## Principal Investigator

<div class="people-grid">
  <div class="person-card">
    <img src="/assets/img/prof_pic.jpg" alt="Kieran Murphy" class="z-depth-1">
    <span class="person-name">Kieran Murphy</span>
    <span class="person-bio">Assistant Professor</span>
  </div>
</div>

**About me:** I am an assistant professor at NJIT. My primary appointment is in the [computer science department](https://cs.njit.edu/), and I have a joint appointment in the [data science department](https://ds.njit.edu/).

**New Jersey Institute of Technology** — Assistant Professor, 2025– <br>
**University of Pennsylvania** — Postdoc, 2021–2025 <br>
**Google Research** — AI Resident, 2019–2021 <br>
**University of Chicago** — PhD (Physics), 2013–2019 <br>
**Lawrence Berkeley National Lab** — Research assistant, 2012–2013 <br>
**UC Berkeley** — BA (Physics, Computer Science), 2009–2013

---

## PhD Students

<div class="people-grid">
{% for person in site.data.people.phd_students %}
  <div class="person-card">
    {% if person.image %}
      <img src="/assets/img/{{ person.image }}" alt="{{ person.name }}" class="z-depth-1">
    {% else %}
      <div class="person-placeholder">&#128100;</div>
    {% endif %}
    <span class="person-name">{% if person.url %}<a href="{{ person.url }}">{{ person.name }}</a>{% else %}{{ person.name }}{% endif %}</span>
    {% if person.bio %}<span class="person-bio">{{ person.bio }}</span>{% endif %}
  </div>
{% endfor %}
</div>

---

## Undergraduates

<div class="people-grid">
{% for person in site.data.people.undergraduates %}
  <div class="person-card">
    {% if person.image %}
      <img src="/assets/img/{{ person.image }}" alt="{{ person.name }}" class="z-depth-1">
    {% else %}
      <div class="person-placeholder">&#128100;</div>
    {% endif %}
    <span class="person-name">{% if person.url %}<a href="{{ person.url }}">{{ person.name }}</a>{% else %}{{ person.name }}{% endif %}</span>
    {% if person.bio %}<span class="person-bio">{{ person.bio }}</span>{% endif %}
  </div>
{% endfor %}
</div>

---

## Alumni

<ul class="alumni-list">
{% for person in site.data.people.alumni %}
  <li>
    {{ person.name }}
    {% if person.note %}<span class="alumni-note">— {{ person.note }}</span>{% endif %}
  </li>
{% endfor %}
</ul>
