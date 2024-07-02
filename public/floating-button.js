// Function to dynamically load the CSS file
function loadCSS(url) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = url;
    document.head.appendChild(link);
}
// Function to dynamically load a JavaScript file
function loadJS(url, callback) {
    const script = document.createElement('script');
    script.src = url;
    script.type = 'text/javascript';
    script.onload = callback;
    document.head.appendChild(script);
}

// Load the floating search CSS file
loadCSS(SearchURL + '/styles.css');
loadJS(SearchURL + '/drawdown.js');
loadJS(SearchURL + '/script.js');

// Create floating button
const floatingButton = document.createElement('div');
floatingButton.id = 'ai-floatingButton';
floatingButton.innerHTML = '<svg style="margin-top: -2px;" fill="#ffffff" height="25px" width="25px" viewBox="0 -0.5 21 21" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><title>search_right [#1507]</title><desc>Created with Sketch.</desc><defs></defs><g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g id="Dribbble-Light-Preview" transform="translate(-179.000000, -280.000000)" fill="#ffffff"><g id="icons" transform="translate(56.000000, 160.000000)"><path d="M128.93985,132.929 L130.42455,134.343 L124.4847,140 L123,138.586 L128.93985,132.929 Z M136.65,132 C133.75515,132 131.4,129.757 131.4,127 C131.4,124.243 133.75515,122 136.65,122 C139.54485,122 141.9,124.243 141.9,127 C141.9,129.757 139.54485,132 136.65,132 L136.65,132 Z M136.65,120 C132.5907,120 129.3,123.134 129.3,127 C129.3,130.866 132.5907,134 136.65,134 C140.7093,134 144,130.866 144,127 C144,123.134 140.7093,120 136.65,120 L136.65,120 Z" id="search_right-[#1507]"></path></g></g></g></svg>';
document.body.appendChild(floatingButton);

// Create modal
const modal = document.createElement('div');
modal.id = 'ai-searchModal';
modal.innerHTML = `
    <div class="ai-modal-content">
        <span class="ai-close-button">&times;</span>
        <div class="ai-search-container">
            <form class="ai-search-box" onsubmit="return false" autocomplete="off">
                <input type="text" autocomplete="off" id="ai-searchInput" placeholder="Enter search term...">
                <button onclick="searchDocuments()">Search</button>
            </form>
        </div>
        <div class="ai-results-container" style="display: none;">
            <div id="ai-loader" class="ai-loader--style" style="display: block">
                <svg version="1.1" id="ai-loader-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
                width="60px" height="60px" viewBox="0 0 40 40" enable-background="new 0 0 40 40" xml:space="preserve">
                <path opacity="0.2" fill="#000" d="M20.201,5.169c-8.254,0-14.946,6.692-14.946,14.946c0,8.255,6.692,14.946,14.946,14.946
                s14.946-6.691,14.946-14.946C35.146,11.861,28.455,5.169,20.201,5.169z M20.201,31.749c-6.425,0-11.634-5.208-11.634-11.634
                c0-6.425,5.209-11.634,11.634-11.634c6.425,0,11.633,5.209,11.633,11.634C31.834,26.541,26.626,31.749,20.201,31.749z"/>
                <path fill="#000" d="M26.013,10.047l1.654-2.866c-2.198-1.272-4.743-2.012-7.466-2.012h0v3.312h0
                C22.32,8.481,24.301,9.057,26.013,10.047z">
                <animateTransform attributeType="xml"
                    attributeName="transform"
                    type="rotate"
                    from="0 20 20"
                    to="360 20 20"
                    dur="0.5s"
                    repeatCount="indefinite"/>
                </path>
                </svg>
            </div>
            <div id="ai-results"></div>
        </div>
    </div>
`;
document.body.appendChild(modal);

// Event listeners
floatingButton.addEventListener('click', () => {
    modal.style.display = 'block';
});

modal.querySelector('.ai-close-button').addEventListener('click', () => {
    modal.style.display = 'none';
});

window.addEventListener('click', (event) => {
    if (event.target == modal) {
        modal.style.display = 'none';
    }
});