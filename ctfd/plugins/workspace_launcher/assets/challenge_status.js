(() => {
  if (window.location.pathname.replace(/\/$/, "") !== "/challenges") return;

  const labelSolvedCards = () => {
    document.querySelectorAll("button.challenge-button.challenge-solved").forEach(button => {
      const name = button.querySelector(".challenge-inner p")?.textContent?.trim();
      if (name) button.setAttribute("aria-label", `${name} — solved by you`);
    });
  };

  labelSolvedCards();
  new MutationObserver(labelSolvedCards).observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class"],
  });
})();
