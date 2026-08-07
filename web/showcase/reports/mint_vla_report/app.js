const sections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll(".toc a")];

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
    });
  },
  { rootMargin: "-18% 0px -66% 0px", threshold: [0.05, 0.25, 0.6] }
);

sections.forEach((section) => observer.observe(section));
