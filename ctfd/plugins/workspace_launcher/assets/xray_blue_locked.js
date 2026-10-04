(() => {
  const lockMessage = "Challenge will be available once X-Ray-Red flag has been submitted";

  const disableLockedBlue = () => {
    if (window.location.pathname.replace(/\/$/, "") !== "/challenges") return;
    document.querySelectorAll("button.challenge-button.tag-xray-blue-locked").forEach(button => {
      button.disabled = true;
      button.setAttribute("aria-disabled", "true");
      const card = button.parentElement;
      if (card) {
        card.classList.add("xray-blue-locked-wrapper");
        card.dataset.lockMessage = lockMessage;
        card.tabIndex = 0;
        card.setAttribute("aria-label", lockMessage);
      }
    });
  };

  disableLockedBlue();
  new MutationObserver(disableLockedBlue).observe(document.body, { childList: true, subtree: true });
})();
