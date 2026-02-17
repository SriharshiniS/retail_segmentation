// Check localStorage for theme preference
document.addEventListener("DOMContentLoaded", () => {
    const currentTheme = localStorage.getItem("theme") || "light";
    document.body.setAttribute("data-theme", currentTheme);

    const toggleButton = document.getElementById("themeToggle");
    toggleButton.addEventListener("click", () => {
        const newTheme = document.body.getAttribute("data-theme") === "light" ? "dark" : "light";
        document.body.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });
});
