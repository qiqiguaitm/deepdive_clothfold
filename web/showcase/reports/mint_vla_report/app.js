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

const filterButtons = [...document.querySelectorAll(".filter-btn")];
const todoItems = [...document.querySelectorAll(".todo-item")];

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const filter = button.dataset.filter;
    filterButtons.forEach((item) => item.classList.toggle("active", item === button));
    todoItems.forEach((item) => {
      item.hidden = filter !== "all" && item.dataset.status !== filter;
    });
  });
});
